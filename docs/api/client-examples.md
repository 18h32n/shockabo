# API Client Examples

This document provides complete, production-ready client examples in multiple programming languages for the ARC Evaluation Framework API.

## Python Client

### Complete Python Client Library

```python
"""
ARC Evaluation Framework API Client
A comprehensive Python client for the ARC evaluation API with authentication,
retry logic, and real-time WebSocket support.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from functools import wraps

import requests
import websockets
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class TaskResult:
    """Result of a task evaluation."""
    submission_id: str
    accuracy: float
    perfect_match: bool
    processing_time_ms: float
    error_category: Optional[str] = None


@dataclass
class ExperimentStatus:
    """Status of a batch experiment."""
    experiment_id: str
    status: str
    progress: float
    completed_tasks: int
    total_tasks: int
    average_accuracy: float
    estimated_completion_time: Optional[datetime] = None


class ARCAPIError(Exception):
    """Base exception for ARC API errors."""
    def __init__(self, message: str, error_code: str = None, status_code: int = None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


class ARCAPIClient:
    """
    Comprehensive client for the ARC Evaluation Framework API.
    
    Features:
    - Automatic authentication and token refresh
    - Exponential backoff retry logic
    - Connection pooling and session management
    - WebSocket support for real-time updates
    - Comprehensive error handling
    """
    
    def __init__(self, base_url: str, user_id: str, timeout: int = 30):
        """
        Initialize the ARC API client.
        
        Args:
            base_url: Base URL of the API (e.g., "http://localhost:8000/api/v1/evaluation")
            user_id: User identifier for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.user_id = user_id
        self.timeout = timeout
        
        # Authentication
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
        
        # HTTP session with connection pooling and retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def authenticate(self) -> bool:
        """
        Authenticate with the API and obtain access tokens.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            response = self.session.post(
                f"{self.base_url}/auth/token",
                json={"user_id": self.user_id},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data.get("refresh_token")
            
            # Calculate token expiration
            expires_in = data.get("expires_in", 1800)  # Default 30 minutes
            self.token_expires = datetime.now() + timedelta(seconds=expires_in - 60)  # 1 minute buffer
            
            # Update session headers
            self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})
            
            self.logger.info(f"Authentication successful for user {self.user_id}")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def _ensure_authenticated(self):
        """Ensure we have a valid access token."""
        if not self.access_token or (self.token_expires and datetime.now() >= self.token_expires):
            if not self.authenticate():
                raise ARCAPIError("Failed to authenticate with API", "AUTH_FAILED")
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and extract JSON data or raise appropriate exceptions.
        
        Args:
            response: HTTP response object
            
        Returns:
            Parsed JSON response data
            
        Raises:
            ARCAPIError: For API errors with detailed information
        """
        try:
            if response.status_code == 200:
                return response.json()
            
            # Try to parse error response
            try:
                error_data = response.json()
                error_code = error_data.get("error_code", "UNKNOWN_ERROR")
                detail = error_data.get("detail", f"HTTP {response.status_code}")
                raise ARCAPIError(detail, error_code, response.status_code)
            except json.JSONDecodeError:
                raise ARCAPIError(f"HTTP {response.status_code}: {response.text}", 
                                status_code=response.status_code)
                
        except requests.RequestException as e:
            raise ARCAPIError(f"Request failed: {e}")
    
    def submit_task(
        self, 
        task_id: str,
        predicted_output: List[List[int]],
        strategy: str = "PATTERN_MATCH",
        confidence_score: float = 0.5,
        attempt_number: int = 1,
        metadata: Dict[str, Any] = None
    ) -> TaskResult:
        """
        Submit a single task for evaluation.
        
        Args:
            task_id: ARC task identifier (e.g., "arc_2024_001")
            predicted_output: 2D grid solution
            strategy: Solving strategy used
            confidence_score: Prediction confidence (0.0-1.0)
            attempt_number: Attempt number (1-2)
            metadata: Additional context data
            
        Returns:
            TaskResult with evaluation metrics
        """
        self._ensure_authenticated()
        
        payload = {
            "task_id": task_id,
            "predicted_output": predicted_output,
            "strategy": strategy,
            "confidence_score": confidence_score,
            "attempt_number": attempt_number,
            "metadata": metadata or {}
        }
        
        response = self.session.post(f"{self.base_url}/submit", json=payload, timeout=self.timeout)
        data = self._handle_response(response)
        
        return TaskResult(
            submission_id=data["submission_id"],
            accuracy=data["accuracy"],
            perfect_match=data["perfect_match"],
            processing_time_ms=data["processing_time_ms"],
            error_category=data.get("error_category")
        )
    
    def submit_batch(
        self,
        evaluations: List[Dict[str, Any]],
        strategy: str = "ENSEMBLE",
        experiment_id: Optional[str] = None,
        parallel_processing: bool = True,
        timeout_seconds: int = 300
    ) -> str:
        """
        Submit multiple tasks for batch evaluation.
        
        Args:
            evaluations: List of task evaluation dictionaries
            strategy: Primary strategy for the batch
            experiment_id: Optional experiment identifier
            parallel_processing: Enable parallel task processing
            timeout_seconds: Maximum processing time
            
        Returns:
            Experiment ID for tracking batch progress
        """
        self._ensure_authenticated()
        
        payload = {
            "evaluations": evaluations,
            "strategy": strategy,
            "experiment_id": experiment_id,
            "parallel_processing": parallel_processing,
            "timeout_seconds": timeout_seconds
        }
        
        response = self.session.post(f"{self.base_url}/evaluate/batch", json=payload, timeout=self.timeout)
        data = self._handle_response(response)
        
        return data["experiment_id"]
    
    def get_experiment_status(self, experiment_id: str) -> ExperimentStatus:
        """
        Get the status of a batch experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            ExperimentStatus with current progress and metrics
        """
        self._ensure_authenticated()
        
        response = self.session.get(f"{self.base_url}/experiments/{experiment_id}/status", timeout=self.timeout)
        data = self._handle_response(response)
        
        estimated_completion = None
        if data.get("estimated_completion_time"):
            estimated_completion = datetime.fromisoformat(data["estimated_completion_time"].replace('Z', '+00:00'))
        
        return ExperimentStatus(
            experiment_id=data["experiment_id"],
            status=data["status"],
            progress=data["progress"],
            completed_tasks=data["completed_tasks"],
            total_tasks=data["total_tasks"],
            average_accuracy=data["average_accuracy"],
            estimated_completion_time=estimated_completion
        )
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get current dashboard metrics and system status.
        
        Returns:
            Dictionary with system metrics and performance data
        """
        self._ensure_authenticated()
        
        response = self.session.get(f"{self.base_url}/dashboard/metrics", timeout=self.timeout)
        return self._handle_response(response)
    
    def get_strategy_performance(self, time_window: str = "24h") -> Dict[str, Any]:
        """
        Get strategy performance analysis.
        
        Args:
            time_window: Analysis window ("1h", "6h", "24h", "7d", "30d")
            
        Returns:
            Dictionary with strategy performance metrics
        """
        self._ensure_authenticated()
        
        params = {"time_window": time_window}
        response = self.session.get(f"{self.base_url}/strategies/performance", params=params, timeout=self.timeout)
        return self._handle_response(response)
    
    async def monitor_experiment(self, experiment_id: str, callback=None) -> ExperimentStatus:
        """
        Monitor an experiment via WebSocket for real-time updates.
        
        Args:
            experiment_id: Experiment to monitor
            callback: Optional callback function for progress updates
            
        Returns:
            Final experiment status when completed
        """
        self._ensure_authenticated()
        
        ws_url = f"{self.base_url.replace('http', 'ws')}/ws?token={self.access_token}"
        
        async with websockets.connect(ws_url) as websocket:
            # Subscribe to experiment updates
            await websocket.send(json.dumps({
                "type": "subscribe_experiment",
                "experiment_id": experiment_id
            }))
            
            final_status = None
            
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("type") == "experiment_progress" and data.get("experiment_id") == experiment_id:
                    if callback:
                        callback(data)
                    
                    # Check if experiment is complete
                    if data.get("progress", 0) >= 1.0:
                        final_status = self.get_experiment_status(experiment_id)
                        break
                
                elif data.get("type") in ["experiment_completed", "experiment_failed"]:
                    if data.get("experiment_id") == experiment_id:
                        final_status = self.get_experiment_status(experiment_id)
                        break
            
            return final_status
    
    def close(self):
        """Close the HTTP session and clean up resources."""
        self.session.close()


# Usage Examples
def example_single_task():
    """Example: Submit single task for evaluation."""
    client = ARCAPIClient("http://localhost:8000/api/v1/evaluation", "demo_user")
    
    try:
        result = client.submit_task(
            task_id="arc_2024_001",
            predicted_output=[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            strategy="PATTERN_MATCH",
            confidence_score=0.92,
            metadata={"pattern_type": "symmetry"}
        )
        
        print(f"Task evaluated: {result.submission_id}")
        print(f"Accuracy: {result.accuracy:.3f}")
        print(f"Perfect match: {result.perfect_match}")
        
    except ARCAPIError as e:
        print(f"API Error [{e.error_code}]: {e}")
    finally:
        client.close()


def example_batch_processing():
    """Example: Process multiple tasks in batch."""
    client = ARCAPIClient("http://localhost:8000/api/v1/evaluation", "demo_user")
    
    try:
        # Prepare batch evaluations
        evaluations = [
            {
                "task_id": "arc_2024_001",
                "predicted_output": [[1, 0], [0, 1]],
                "confidence": 0.9,
                "attempt_number": 1
            },
            {
                "task_id": "arc_2024_002",
                "predicted_output": [[2, 3], [4, 5]],
                "confidence": 0.8,
                "attempt_number": 1
            }
        ]
        
        # Submit batch
        experiment_id = client.submit_batch(
            evaluations=evaluations,
            strategy="ENSEMBLE",
            experiment_id="demo_experiment_001"
        )
        
        print(f"Batch submitted: {experiment_id}")
        
        # Monitor progress
        while True:
            status = client.get_experiment_status(experiment_id)
            print(f"Progress: {status.progress:.1%} ({status.completed_tasks}/{status.total_tasks})")
            
            if status.status in ["COMPLETED", "FAILED"]:
                print(f"Final status: {status.status}")
                print(f"Average accuracy: {status.average_accuracy:.3f}")
                break
            
            time.sleep(5)  # Poll every 5 seconds
            
    except ARCAPIError as e:
        print(f"API Error [{e.error_code}]: {e}")
    finally:
        client.close()


async def example_realtime_monitoring():
    """Example: Real-time experiment monitoring via WebSocket."""
    client = ARCAPIClient("http://localhost:8000/api/v1/evaluation", "demo_user")
    
    try:
        # Start batch experiment
        evaluations = [{"task_id": f"arc_2024_{i:03d}", "predicted_output": [[i]], "confidence": 0.5} 
                      for i in range(1, 21)]
        
        experiment_id = client.submit_batch(evaluations, strategy="PATTERN_MATCH")
        print(f"Started experiment: {experiment_id}")
        
        # Monitor via WebSocket
        def progress_callback(update):
            print(f"Progress update: {update['progress']:.1%} - Accuracy: {update['current_accuracy']:.3f}")
        
        final_status = await client.monitor_experiment(experiment_id, progress_callback)
        print(f"Experiment completed with accuracy: {final_status.average_accuracy:.3f}")
        
    except ARCAPIError as e:
        print(f"API Error [{e.error_code}]: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    print("=== Single Task Example ===")
    example_single_task()
    
    print("\n=== Batch Processing Example ===")
    example_batch_processing()
    
    print("\n=== Real-time Monitoring Example ===")
    asyncio.run(example_realtime_monitoring())
```

## JavaScript/Node.js Client

### Complete JavaScript Client

```javascript
/**
 * ARC Evaluation Framework API Client for JavaScript/Node.js
 * 
 * Features:
 * - Promise-based API with async/await support
 * - Automatic authentication and token refresh
 * - WebSocket support for real-time updates
 * - Comprehensive error handling
 * - Rate limiting and retry logic
 */

const axios = require('axios');
const WebSocket = require('ws');
const EventEmitter = require('events');

class ARCAPIError extends Error {
    constructor(message, errorCode = null, statusCode = null) {
        super(message);
        this.name = 'ARCAPIError';
        this.errorCode = errorCode;
        this.statusCode = statusCode;
    }
}

class ARCAPIClient extends EventEmitter {
    /**
     * Initialize the ARC API client
     * @param {string} baseUrl - Base URL of the API
     * @param {string} userId - User identifier for authentication
     * @param {Object} options - Additional options
     */
    constructor(baseUrl, userId, options = {}) {
        super();
        
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.userId = userId;
        this.timeout = options.timeout || 30000;
        
        // Authentication
        this.accessToken = null;
        this.refreshToken = null;
        this.tokenExpires = null;
        
        // HTTP client with interceptors
        this.httpClient = axios.create({
            baseURL: this.baseUrl,
            timeout: this.timeout,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        this._setupInterceptors();
    }
    
    /**
     * Set up request/response interceptors for authentication and error handling
     * @private
     */
    _setupInterceptors() {
        // Request interceptor for authentication
        this.httpClient.interceptors.request.use(async (config) => {
            await this._ensureAuthenticated();
            if (this.accessToken) {
                config.headers.Authorization = `Bearer ${this.accessToken}`;
            }
            return config;
        });
        
        // Response interceptor for error handling
        this.httpClient.interceptors.response.use(
            response => response,
            error => {
                if (error.response) {
                    const { data, status } = error.response;
                    const errorCode = data?.error_code || 'UNKNOWN_ERROR';
                    const detail = data?.detail || `HTTP ${status}`;
                    throw new ARCAPIError(detail, errorCode, status);
                }
                throw new ARCAPIError(error.message);
            }
        );
    }
    
    /**
     * Authenticate with the API and obtain access tokens
     * @returns {Promise<boolean>} True if authentication successful
     */
    async authenticate() {
        try {
            const response = await axios.post(`${this.baseUrl}/auth/token`, {
                user_id: this.userId
            });
            
            const { access_token, refresh_token, expires_in = 1800 } = response.data;
            
            this.accessToken = access_token;
            this.refreshToken = refresh_token;
            
            // Calculate token expiration with 1 minute buffer
            this.tokenExpires = new Date(Date.now() + (expires_in - 60) * 1000);
            
            console.log(`Authentication successful for user ${this.userId}`);
            return true;
            
        } catch (error) {
            console.error('Authentication failed:', error.message);
            return false;
        }
    }
    
    /**
     * Ensure we have a valid access token
     * @private
     */
    async _ensureAuthenticated() {
        if (!this.accessToken || (this.tokenExpires && new Date() >= this.tokenExpires)) {
            const success = await this.authenticate();
            if (!success) {
                throw new ARCAPIError('Failed to authenticate with API', 'AUTH_FAILED');
            }
        }
    }
    
    /**
     * Submit a single task for evaluation
     * @param {Object} taskData - Task submission data
     * @returns {Promise<Object>} Task evaluation result
     */
    async submitTask(taskData) {
        const {
            taskId,
            predictedOutput,
            strategy = 'PATTERN_MATCH',
            confidenceScore = 0.5,
            attemptNumber = 1,
            metadata = {}
        } = taskData;
        
        const payload = {
            task_id: taskId,
            predicted_output: predictedOutput,
            strategy: strategy,
            confidence_score: confidenceScore,
            attempt_number: attemptNumber,
            metadata: metadata
        };
        
        const response = await this.httpClient.post('/submit', payload);
        return response.data;
    }
    
    /**
     * Submit multiple tasks for batch evaluation
     * @param {Object} batchData - Batch submission data
     * @returns {Promise<string>} Experiment ID
     */
    async submitBatch(batchData) {
        const {
            evaluations,
            strategy = 'ENSEMBLE',
            experimentId = null,
            parallelProcessing = true,
            timeoutSeconds = 300
        } = batchData;
        
        const payload = {
            evaluations,
            strategy,
            experiment_id: experimentId,
            parallel_processing: parallelProcessing,
            timeout_seconds: timeoutSeconds
        };
        
        const response = await this.httpClient.post('/evaluate/batch', payload);
        return response.data.experiment_id;
    }
    
    /**
     * Get the status of a batch experiment
     * @param {string} experimentId - Experiment identifier
     * @returns {Promise<Object>} Experiment status
     */
    async getExperimentStatus(experimentId) {
        const response = await this.httpClient.get(`/experiments/${experimentId}/status`);
        return response.data;
    }
    
    /**
     * Get current dashboard metrics
     * @returns {Promise<Object>} Dashboard metrics
     */
    async getDashboardMetrics() {
        const response = await this.httpClient.get('/dashboard/metrics');
        return response.data;
    }
    
    /**
     * Get strategy performance analysis
     * @param {string} timeWindow - Analysis window ("1h", "6h", "24h", "7d", "30d")
     * @returns {Promise<Object>} Strategy performance data
     */
    async getStrategyPerformance(timeWindow = '24h') {
        const response = await this.httpClient.get('/strategies/performance', {
            params: { time_window: timeWindow }
        });
        return response.data;
    }
    
    /**
     * Create WebSocket connection for real-time updates
     * @returns {Promise<WebSocket>} WebSocket connection
     */
    async createWebSocketConnection() {
        await this._ensureAuthenticated();
        
        const wsUrl = this.baseUrl.replace(/^http/, 'ws') + `/ws?token=${this.accessToken}`;
        const ws = new WebSocket(wsUrl);
        
        return new Promise((resolve, reject) => {
            ws.on('open', () => {
                console.log('WebSocket connection established');
                resolve(ws);
            });
            
            ws.on('error', (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            });
            
            ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data);
                    this.emit('message', message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            });
            
            ws.on('close', (code, reason) => {
                console.log(`WebSocket connection closed: ${code} ${reason}`);
                this.emit('disconnect', { code, reason });
            });
        });
    }
    
    /**
     * Monitor experiment progress via WebSocket
     * @param {string} experimentId - Experiment to monitor
     * @returns {Promise<Object>} Final experiment status
     */
    async monitorExperiment(experimentId) {
        const ws = await this.createWebSocketConnection();
        
        // Subscribe to experiment updates
        ws.send(JSON.stringify({
            type: 'subscribe_experiment',
            experiment_id: experimentId
        }));
        
        return new Promise((resolve, reject) => {
            const handleMessage = (message) => {
                if (message.type === 'experiment_progress' && message.experiment_id === experimentId) {
                    this.emit('experimentProgress', message);
                    
                    if (message.progress >= 1.0) {
                        this._finishMonitoring(experimentId, ws, resolve, reject);
                    }
                } else if (message.type === 'experiment_completed' && message.experiment_id === experimentId) {
                    this._finishMonitoring(experimentId, ws, resolve, reject);
                } else if (message.type === 'experiment_failed' && message.experiment_id === experimentId) {
                    this._finishMonitoring(experimentId, ws, resolve, reject);
                }
            };
            
            this.on('message', handleMessage);
            
            // Handle WebSocket errors
            this.on('disconnect', ({ code, reason }) => {
                this.removeListener('message', handleMessage);
                reject(new Error(`WebSocket disconnected: ${code} ${reason}`));
            });
        });
    }
    
    /**
     * Finish experiment monitoring and get final status
     * @private
     */
    async _finishMonitoring(experimentId, ws, resolve, reject) {
        try {
            const finalStatus = await this.getExperimentStatus(experimentId);
            ws.close();
            resolve(finalStatus);
        } catch (error) {
            ws.close();
            reject(error);
        }
    }
    
    /**
     * Utility method to wait for experiment completion with polling
     * @param {string} experimentId - Experiment identifier
     * @param {number} pollInterval - Polling interval in milliseconds
     * @returns {Promise<Object>} Final experiment status
     */
    async waitForExperiment(experimentId, pollInterval = 5000) {
        while (true) {
            const status = await this.getExperimentStatus(experimentId);
            
            this.emit('experimentStatus', status);
            
            if (['COMPLETED', 'FAILED', 'CANCELLED'].includes(status.status)) {
                return status;
            }
            
            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }
    }
}

// Usage Examples

async function exampleSingleTask() {
    const client = new ARCAPIClient('http://localhost:8000/api/v1/evaluation', 'demo_user');
    
    try {
        const result = await client.submitTask({
            taskId: 'arc_2024_001',
            predictedOutput: [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            strategy: 'PATTERN_MATCH',
            confidenceScore: 0.92,
            metadata: { patternType: 'symmetry' }
        });
        
        console.log('Task evaluated:', result.submission_id);
        console.log('Accuracy:', result.accuracy.toFixed(3));
        console.log('Perfect match:', result.perfect_match);
        
    } catch (error) {
        console.error(`API Error [${error.errorCode}]:`, error.message);
    }
}

async function exampleBatchProcessing() {
    const client = new ARCAPIClient('http://localhost:8000/api/v1/evaluation', 'demo_user');
    
    try {
        const evaluations = [
            {
                task_id: 'arc_2024_001',
                predicted_output: [[1, 0], [0, 1]],
                confidence: 0.9,
                attempt_number: 1
            },
            {
                task_id: 'arc_2024_002',
                predicted_output: [[2, 3], [4, 5]],
                confidence: 0.8,
                attempt_number: 1
            }
        ];
        
        const experimentId = await client.submitBatch({
            evaluations,
            strategy: 'ENSEMBLE',
            experimentId: 'demo_experiment_001'
        });
        
        console.log('Batch submitted:', experimentId);
        
        // Monitor with polling
        client.on('experimentStatus', (status) => {
            const progress = (status.progress * 100).toFixed(1);
            console.log(`Progress: ${progress}% (${status.completed_tasks}/${status.total_tasks})`);
        });
        
        const finalStatus = await client.waitForExperiment(experimentId);
        console.log('Final status:', finalStatus.status);
        console.log('Average accuracy:', finalStatus.average_accuracy.toFixed(3));
        
    } catch (error) {
        console.error(`API Error [${error.errorCode}]:`, error.message);
    }
}

async function exampleRealtimeMonitoring() {
    const client = new ARCAPIClient('http://localhost:8000/api/v1/evaluation', 'demo_user');
    
    try {
        // Start batch experiment
        const evaluations = Array.from({ length: 20 }, (_, i) => ({
            task_id: `arc_2024_${String(i + 1).padStart(3, '0')}`,
            predicted_output: [[i]],
            confidence: 0.5
        }));
        
        const experimentId = await client.submitBatch({
            evaluations,
            strategy: 'PATTERN_MATCH'
        });
        
        console.log('Started experiment:', experimentId);
        
        // Monitor via WebSocket
        client.on('experimentProgress', (update) => {
            const progress = (update.progress * 100).toFixed(1);
            const accuracy = update.current_accuracy.toFixed(3);
            console.log(`Progress update: ${progress}% - Accuracy: ${accuracy}`);
        });
        
        const finalStatus = await client.monitorExperiment(experimentId);
        console.log('Experiment completed with accuracy:', finalStatus.average_accuracy.toFixed(3));
        
    } catch (error) {
        console.error(`API Error [${error.errorCode}]:`, error.message);
    }
}

// Export for use as module
module.exports = { ARCAPIClient, ARCAPIError };

// Run examples if this file is executed directly
if (require.main === module) {
    console.log('=== Single Task Example ===');
    exampleSingleTask()
        .then(() => console.log('\n=== Batch Processing Example ==='))
        .then(() => exampleBatchProcessing())
        .then(() => console.log('\n=== Real-time Monitoring Example ==='))
        .then(() => exampleRealtimeMonitoring())
        .catch(console.error);
}
```

## cURL Examples

### Authentication

```bash
# Get authentication token
curl -X POST "http://localhost:8000/api/v1/evaluation/auth/token" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "demo_user"}'

# Save token for subsequent requests
export TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

### Single Task Submission

```bash
# Submit single task
curl -X POST "http://localhost:8000/api/v1/evaluation/submit" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "arc_2024_001",
       "predicted_output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
       "strategy": "PATTERN_MATCH",
       "confidence_score": 0.92,
       "attempt_number": 1,
       "metadata": {
         "pattern_type": "symmetry",
         "processing_time_ms": 1450
       }
     }'
```

### Batch Processing

```bash
# Submit batch evaluation
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate/batch" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "evaluations": [
         {
           "task_id": "arc_2024_001",
           "predicted_output": [[1, 0], [0, 1]],
           "confidence": 0.9,
           "attempt_number": 1
         },
         {
           "task_id": "arc_2024_002",
           "predicted_output": [[2, 3], [4, 5]],
           "confidence": 0.8,
           "attempt_number": 1
         }
       ],
       "strategy": "ENSEMBLE",
       "experiment_id": "demo_batch_001",
       "parallel_processing": true,
       "timeout_seconds": 300
     }'
```

### Monitoring and Status

```bash
# Get experiment status
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/evaluation/experiments/demo_batch_001/status"

# Get dashboard metrics
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/evaluation/dashboard/metrics"

# Get strategy performance (24 hour window)
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/evaluation/strategies/performance?time_window=24h"

# Get connection pool statistics
curl -H "Authorization: Bearer $TOKEN" \
     "http://localhost:8000/api/v1/evaluation/dashboard/connection-pool/stats"
```

## WebSocket Examples

### JavaScript WebSocket Client

```html
<!DOCTYPE html>
<html>
<head>
    <title>ARC Evaluation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div id="dashboard">
        <h1>ARC Evaluation Dashboard</h1>
        <div id="status">Connecting...</div>
        <div id="metrics"></div>
        <canvas id="accuracyChart" width="400" height="200"></canvas>
    </div>

    <script>
        class ARCDashboard {
            constructor(token) {
                this.token = token;
                this.ws = null;
                this.chart = null;
                this.accuracyData = [];
                this.init();
            }
            
            init() {
                this.connect();
                this.setupChart();
            }
            
            connect() {
                const wsUrl = `ws://localhost:8000/api/v1/evaluation/ws?token=${this.token}`;
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    document.getElementById('status').textContent = 'Connected';
                    console.log('WebSocket connected');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    document.getElementById('status').textContent = 'Connection Error';
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket closed');
                    document.getElementById('status').textContent = 'Disconnected';
                    // Reconnect after 5 seconds
                    setTimeout(() => this.connect(), 5000);
                };
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'connection_established':
                        console.log('Dashboard connection established');
                        break;
                        
                    case 'dashboard_update':
                        this.updateDashboard(data.data);
                        break;
                        
                    case 'task_submitted':
                        this.handleTaskSubmission(data);
                        break;
                        
                    case 'experiment_progress':
                        this.handleExperimentProgress(data);
                        break;
                        
                    default:
                        console.log('Unknown message type:', data.type);
                }
            }
            
            updateDashboard(metrics) {
                document.getElementById('metrics').innerHTML = `
                    <h3>System Metrics</h3>
                    <p>Active Experiments: ${metrics.active_experiments}</p>
                    <p>Tasks Processed: ${metrics.tasks_processed}</p>
                    <p>Average Accuracy: ${(metrics.average_accuracy * 100).toFixed(1)}%</p>
                    <p>CPU Usage: ${metrics.resource_utilization.cpu.toFixed(1)}%</p>
                    <p>Memory Usage: ${metrics.resource_utilization.memory.toFixed(1)}%</p>
                `;
                
                // Update accuracy chart
                this.accuracyData.push({
                    time: new Date().toLocaleTimeString(),
                    accuracy: metrics.average_accuracy
                });
                
                // Keep only last 20 data points
                if (this.accuracyData.length > 20) {
                    this.accuracyData.shift();
                }
                
                this.updateChart();
            }
            
            handleTaskSubmission(data) {
                console.log(`Task ${data.task_id} submitted with accuracy: ${data.accuracy}`);
            }
            
            handleExperimentProgress(data) {
                console.log(`Experiment ${data.experiment_id}: ${(data.progress * 100).toFixed(1)}% complete`);
            }
            
            setupChart() {
                const ctx = document.getElementById('accuracyChart').getContext('2d');
                this.chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Average Accuracy',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1
                            }
                        }
                    }
                });
            }
            
            updateChart() {
                this.chart.data.labels = this.accuracyData.map(d => d.time);
                this.chart.data.datasets[0].data = this.accuracyData.map(d => d.accuracy);
                this.chart.update('none'); // Update without animation for real-time feel
            }
            
            subscribeToExperiment(experimentId) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        type: 'subscribe_experiment',
                        experiment_id: experimentId
                    }));
                }
            }
            
            disconnect() {
                if (this.ws) {
                    this.ws.close();
                }
            }
        }
        
        // Initialize dashboard (replace with your actual token)
        const dashboard = new ARCDashboard('your_jwt_token_here');
        
        // Example: Subscribe to specific experiment
        // dashboard.subscribeToExperiment('exp_demo_001');
    </script>
</body>
</html>
```

## Go Client Example

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
    "net/url"
    "time"
)

// ARCAPIClient represents a client for the ARC Evaluation API
type ARCAPIClient struct {
    BaseURL     string
    UserID      string
    AccessToken string
    HTTPClient  *http.Client
}

// TaskSubmission represents a task submission request
type TaskSubmission struct {
    TaskID          string              `json:"task_id"`
    PredictedOutput [][]int            `json:"predicted_output"`
    Strategy        string              `json:"strategy"`
    ConfidenceScore float64             `json:"confidence_score"`
    AttemptNumber   int                 `json:"attempt_number"`
    Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// TaskResult represents the response from task submission
type TaskResult struct {
    SubmissionID     string  `json:"submission_id"`
    Accuracy         float64 `json:"accuracy"`
    PerfectMatch     bool    `json:"perfect_match"`
    ProcessingTimeMs float64 `json:"processing_time_ms"`
    ErrorCategory    *string `json:"error_category,omitempty"`
}

// NewARCAPIClient creates a new API client
func NewARCAPIClient(baseURL, userID string) *ARCAPIClient {
    return &ARCAPIClient{
        BaseURL: baseURL,
        UserID:  userID,
        HTTPClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

// Authenticate obtains an access token
func (c *ARCAPIClient) Authenticate() error {
    authData := map[string]string{"user_id": c.UserID}
    jsonData, _ := json.Marshal(authData)
    
    resp, err := c.HTTPClient.Post(
        c.BaseURL+"/auth/token",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("authentication failed with status: %d", resp.StatusCode)
    }
    
    var authResp struct {
        AccessToken string `json:"access_token"`
    }
    
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return err
    }
    
    if err := json.Unmarshal(body, &authResp); err != nil {
        return err
    }
    
    c.AccessToken = authResp.AccessToken
    return nil
}

// SubmitTask submits a single task for evaluation
func (c *ARCAPIClient) SubmitTask(submission TaskSubmission) (*TaskResult, error) {
    if c.AccessToken == "" {
        if err := c.Authenticate(); err != nil {
            return nil, err
        }
    }
    
    jsonData, err := json.Marshal(submission)
    if err != nil {
        return nil, err
    }
    
    req, err := http.NewRequest("POST", c.BaseURL+"/submit", bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, err
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+c.AccessToken)
    
    resp, err := c.HTTPClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
    }
    
    var result TaskResult
    if err := json.Unmarshal(body, &result); err != nil {
        return nil, err
    }
    
    return &result, nil
}

// GetDashboardMetrics retrieves current dashboard metrics
func (c *ARCAPIClient) GetDashboardMetrics() (map[string]interface{}, error) {
    if c.AccessToken == "" {
        if err := c.Authenticate(); err != nil {
            return nil, err
        }
    }
    
    req, err := http.NewRequest("GET", c.BaseURL+"/dashboard/metrics", nil)
    if err != nil {
        return nil, err
    }
    
    req.Header.Set("Authorization", "Bearer "+c.AccessToken)
    
    resp, err := c.HTTPClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
    }
    
    var metrics map[string]interface{}
    if err := json.Unmarshal(body, &metrics); err != nil {
        return nil, err
    }
    
    return metrics, nil
}

// Example usage
func main() {
    client := NewARCAPIClient("http://localhost:8000/api/v1/evaluation", "demo_user")
    
    // Submit a task
    submission := TaskSubmission{
        TaskID:          "arc_2024_001",
        PredictedOutput: [][]int{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}},
        Strategy:        "PATTERN_MATCH",
        ConfidenceScore: 0.92,
        AttemptNumber:   1,
        Metadata:        map[string]interface{}{"pattern_type": "symmetry"},
    }
    
    result, err := client.SubmitTask(submission)
    if err != nil {
        fmt.Printf("Error submitting task: %v\n", err)
        return
    }
    
    fmt.Printf("Task evaluated: %s\n", result.SubmissionID)
    fmt.Printf("Accuracy: %.3f\n", result.Accuracy)
    fmt.Printf("Perfect match: %t\n", result.PerfectMatch)
    
    // Get dashboard metrics
    metrics, err := client.GetDashboardMetrics()
    if err != nil {
        fmt.Printf("Error getting metrics: %v\n", err)
        return
    }
    
    fmt.Printf("Dashboard metrics: %+v\n", metrics)
}
```

## Testing and Development

### Integration Testing

```python
import pytest
import asyncio
from arc_api_client import ARCAPIClient, ARCAPIError

class TestARCAPI:
    @pytest.fixture
    def client(self):
        return ARCAPIClient("http://localhost:8000/api/v1/evaluation", "test_user")
    
    def test_authentication(self, client):
        """Test API authentication"""
        assert client.authenticate() == True
        assert client.access_token is not None
    
    def test_single_task_submission(self, client):
        """Test single task submission"""
        result = client.submit_task(
            task_id="arc_test_001",
            predicted_output=[[1, 0], [0, 1]],
            strategy="DIRECT_SOLVE",
            confidence_score=0.8
        )
        
        assert result.submission_id is not None
        assert 0 <= result.accuracy <= 1
        assert isinstance(result.perfect_match, bool)
    
    def test_batch_processing(self, client):
        """Test batch evaluation"""
        evaluations = [
            {
                "task_id": "arc_test_001",
                "predicted_output": [[1, 0], [0, 1]],
                "confidence": 0.8,
                "attempt_number": 1
            }
        ]
        
        experiment_id = client.submit_batch(
            evaluations=evaluations,
            strategy="ENSEMBLE"
        )
        
        assert experiment_id is not None
        
        # Wait for completion
        final_status = client.wait_for_experiment(experiment_id, poll_interval=1000)
        assert final_status.status in ["COMPLETED", "FAILED"]
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, client):
        """Test WebSocket connectivity"""
        ws = await client.create_websocket_connection()
        assert ws is not None
        
        # Send ping
        ws.send('{"type": "ping"}')
        
        # Should receive pong
        response = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(response)
        assert data["type"] == "pong"
        
        ws.close()
    
    def test_error_handling(self, client):
        """Test API error handling"""
        with pytest.raises(ARCAPIError) as exc_info:
            client.submit_task(
                task_id="invalid_task_id_format",
                predicted_output=[[1, 0], [0, 1]],
                strategy="DIRECT_SOLVE"
            )
        
        assert exc_info.value.error_code == "INVALID_TASK_ID"
```

### Load Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from arc_api_client import ARCAPIClient

async def load_test_submissions(num_clients=10, tasks_per_client=50):
    """Load test with concurrent task submissions"""
    
    async def submit_tasks_for_client(client_id):
        client = ARCAPIClient("http://localhost:8000/api/v1/evaluation", f"load_test_user_{client_id}")
        
        results = []
        start_time = time.time()
        
        for i in range(tasks_per_client):
            try:
                result = client.submit_task(
                    task_id=f"arc_load_test_{i:03d}",
                    predicted_output=[[i % 10] * 3] * 3,
                    strategy="DIRECT_SOLVE",
                    confidence_score=0.5 + (i % 50) / 100
                )
                results.append(result)
                
            except Exception as e:
                print(f"Client {client_id}, Task {i}: Error - {e}")
        
        end_time = time.time()
        client.close()
        
        return {
            'client_id': client_id,
            'tasks_completed': len(results),
            'duration': end_time - start_time,
            'avg_accuracy': sum(r.accuracy for r in results) / len(results) if results else 0
        }
    
    # Run concurrent clients
    tasks = [submit_tasks_for_client(i) for i in range(num_clients)]
    results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    total_tasks = sum(r['tasks_completed'] for r in results)
    total_duration = max(r['duration'] for r in results)
    avg_accuracy = sum(r['avg_accuracy'] for r in results) / len(results)
    
    print(f"Load Test Results:")
    print(f"Total tasks: {total_tasks}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Tasks per second: {total_tasks / total_duration:.2f}")
    print(f"Average accuracy: {avg_accuracy:.3f}")
    
    return results

if __name__ == "__main__":
    asyncio.run(load_test_submissions())
```

This comprehensive set of client examples provides production-ready code in multiple languages with proper error handling, authentication, and real-time monitoring capabilities.