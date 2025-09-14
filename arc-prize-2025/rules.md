# ARC Prize 2025: Competition Rules and Requirements

## Competition Overview

The ARC Prize 2025 is a $825,000+ competition challenging teams to create AI systems capable of novel reasoning to achieve artificial general intelligence (AGI) by reaching 85% accuracy on the ARC-AGI-2 private evaluation dataset.

## Timeline and Deadlines

### Key Dates
- **Competition Launch**: March 26, 2025
- **Final Submission Deadline**: November 3, 2025 (11:59 PM UTC)
- **Paper Submission Deadline**: November 9, 2025 (11:59 PM UTC)
- **Winners Announced**: December 5, 2025

### Important Notes
- All submissions must be completed by the final deadline
- Late submissions will not be accepted under any circumstances
- Teams must meet all open source requirements before final scoring

## Eligibility Requirements

### Team Composition
- **Individual or Team**: Both individual participants and teams are allowed
- **Team Size**: No explicit limit on team size
- **Registration**: Must register on Kaggle with valid account
- **Geographic**: Open to participants worldwide (subject to applicable laws)

### Participation Restrictions
- Kaggle staff and ARC Prize organizers may be restricted from winning prizes
- Standard Kaggle competition eligibility rules apply
- Participants must comply with all applicable laws and regulations

## Technical Requirements and Constraints

### Hardware and Runtime Limits
- **Maximum Runtime**: 12 hours total (CPU + GPU time combined)
- **Hardware Platform**: Kaggle L4x4 GPUs with 96GB GPU memory
- **Internet Access**: **NO INTERNET ACCESS** during evaluation
- **Storage**: Limited to Kaggle notebook environment constraints

### Allowed External Resources
- **Pre-trained Models**: Allowed if freely available and properly licensed
- **External Data**: Permitted if freely available and legally usable
- **Code Libraries**: Standard machine learning and programming libraries
- **Offline Resources**: Any resources that can be included in submission without internet

### Prohibited Resources
- **Internet Access**: No web requests, API calls, or external connections during evaluation
- **Dynamic Downloads**: Cannot download models, data, or code during runtime
- **Proprietary Data**: Cannot use proprietary or licensed datasets
- **External Services**: No cloud services, external APIs, or third-party processing

## Submission Format and Requirements

### Submission File Structure
Teams must submit a complete Kaggle notebook containing:
1. **Code**: All source code required for solution
2. **Models**: All trained models and weights (if applicable)
3. **Dependencies**: All required libraries and dependencies
4. **Output**: Submission.json file with predictions

### Output Format Specification
The submission.json file must follow this exact format:

```json
{
  "task_001": [
    {
      "attempt_1": [[int, int, ...], [int, int, ...], ...],
      "attempt_2": [[int, int, ...], [int, int, ...], ...]
    }
  ],
  "task_002": [
    {
      "attempt_1": [[int, int, ...], [int, int, ...], ...],
      "attempt_2": [[int, int, ...], [int, int, ...], ...]
    }
  ]
  // ... for all 120 private evaluation tasks
}
```

### Submission Requirements
- **Complete Predictions**: Must provide predictions for ALL private evaluation tasks
- **Two Attempts**: Exactly 2 prediction attempts per task required
- **Correct Format**: Grid dimensions must match expected output exactly
- **Valid Values**: Only integers 0-9 allowed (representing colors)
- **File Size**: Must comply with Kaggle submission limits

## Scoring Methodology

### Primary Scoring
- **Metric**: Percentage of correct predictions on private evaluation set
- **Calculation**: (Number of correct task outputs) / (Total task outputs)
- **Accuracy Requirement**: 100% pixel-perfect match required for scoring
- **Attempts**: Score is based on best of 2 attempts per task

### Scoring Example
- Private evaluation set: 120 tasks
- Task with correct attempt 1: +1 point
- Task with correct attempt 2: +1 point  
- Task with both attempts incorrect: +0 points
- Final score: (Correct tasks) / 120 = Percentage accuracy

### Tie-Breaking
In case of identical scores:
1. **Submission Time**: Earlier submission time preferred
2. **Code Quality**: Judged by competition organizers
3. **Approach Novelty**: More innovative approaches preferred

## Prize Structure and Distribution

### Grand Prize: $700,000
- **Eligibility**: Teams scoring ≥85% on private evaluation set
- **Recipients**: Top 5 teams meeting the 85% threshold
- **Distribution**: Equal split among qualifying teams ($140,000 each if 5 teams qualify)
- **Rollover**: If no team reaches 85%, prize rolls over to next year

### Paper Awards: $75,000 Total
- **First Place**: $50,000 (Best paper overall)
- **Second Place**: $20,000 (Second best paper)
- **Third Place**: $5,000 (Third best paper)
- **Evaluation Criteria**:
  - Accuracy of approach
  - Universality and generalization
  - Progress toward AGI
  - Theoretical contribution
  - Completeness of solution
  - Novelty of ideas

### Top Score Awards: $50,000 Total
- **First Place**: $25,000 (Highest accuracy score)
- **Second Place**: $5,000 (Second highest score)
- **Third Place**: $5,000 (Third highest score)  
- **Fourth Place**: $5,000 (Fourth highest score)
- **Fifth Place**: $5,000 (Fifth highest score)
- **Independent**: Awarded regardless of 85% threshold

### Additional Prizes: $175,000
- Reserved for potential additional prizes to be announced
- May include innovation awards, community contribution awards, etc.

## Open Source Requirements (CRITICAL)

### Pre-Submission Open Source Mandate
- **Timing**: Teams must open source their solutions **BEFORE** seeing final private scores
- **Verification**: Kaggle will verify open source compliance before final scoring
- **No Exceptions**: Failure to open source disqualifies from all prizes

### Code Licensing Requirements
- **Submitter Code**: Must be released under public domain licenses (CC0 or MIT-0)
- **Third-Party Code**: Must use licenses allowing public sharing (Apache-2.0, GPLv3+, etc.)
- **Documentation**: Must include clear documentation and usage instructions
- **Reproducibility**: Code must be sufficient to reproduce results

### Open Source Content Requirements
- **Complete Solution**: All code used in final submission
- **Model Weights**: All trained models and parameters (if applicable)
- **Training Code**: Code used for model training and development
- **Preprocessing**: All data preprocessing and pipeline code
- **Dependencies**: Clear specification of all requirements and versions

### Publication Platform
- **GitHub**: Preferred platform for code repository
- **Documentation**: Must include comprehensive README
- **License File**: Must include appropriate license file
- **Contact**: Must include contact information for questions

## Competition Philosophy and Ethics

### Primary Mission
The ARC Prize exists to **accelerate open AGI progress** by:
- Encouraging breakthrough research in artificial general intelligence
- Making cutting-edge solutions freely available to the research community
- Fostering collaboration and knowledge sharing
- Advancing the scientific understanding of intelligence

### Fair Play Requirements
- **No Cheating**: Any form of cheating results in immediate disqualification
- **Original Work**: Solutions must be original work of the team
- **Proper Attribution**: Must properly cite and attribute third-party work
- **Data Integrity**: Cannot manipulate or corrupt competition data

### Community Guidelines
- **Respectful Collaboration**: Encourage respectful discussion and collaboration
- **Knowledge Sharing**: Share insights and learnings with community
- **Help Others**: Assist other participants when possible
- **Follow Guidelines**: Adhere to Kaggle and ARC Prize community standards

## Evaluation Process

### Evaluation Phases
1. **Local Testing**: Teams test on public evaluation set (120 tasks)
2. **Leaderboard**: Semi-private evaluation set (120 tasks) for public rankings
3. **Final Scoring**: Private evaluation set (120 tasks) for final ranking

### Private Evaluation
- **One-Time Scoring**: Private set scored only once at competition end
- **No Feedback**: No intermediate feedback on private performance
- **Final Results**: Determines all prize winners and final rankings

### Verification Process
1. **Code Review**: Manual review of submission notebooks
2. **Reproducibility**: Verification that code produces submitted results  
3. **Resource Compliance**: Confirm adherence to runtime and resource limits
4. **Open Source**: Verify open source requirements met before scoring

## Data Usage Rules

### Training Data Usage
- **Public Training Set**: 1,000 tasks available for unrestricted use
- **Public Evaluation Set**: 120 tasks available for validation
- **Prohibited**: Cannot use semi-private or private task outputs for training

### External Data Integration
- **Allowed**: Any freely available, legally usable data
- **Attribution**: Must properly document and attribute external data sources
- **Licensing**: Must comply with all data licensing requirements
- **No Leakage**: Cannot use data that contains ARC-AGI task solutions

## Technical Support and Resources

### Official Resources
- **Competition Page**: https://www.kaggle.com/competitions/arc-prize-2025
- **ARC Prize Website**: https://arcprize.org/
- **Official Guide**: https://arcprize.org/guide
- **GitHub Repository**: https://github.com/arcprize/ARC-AGI-2

### Community Support
- **Discord Server**: https://discord.gg/9b77dPAmcA
- **Kaggle Discussions**: Competition discussion forum
- **Email Support**: team@arcprize.org for technical issues

### Available Tools
- **Task Viewer**: Web-based visualization at arcprize.org
- **Starter Notebooks**: Community-contributed baseline solutions
- **Visualization Tools**: Community-created analysis and visualization code

## Rule Violations and Enforcement

### Minor Violations
- **Warning**: First offense may result in warning
- **Correction**: Opportunity to correct minor technical issues
- **Monitoring**: Continued monitoring of compliance

### Major Violations
- **Disqualification**: Immediate removal from competition
- **Prize Forfeiture**: Loss of eligibility for all prizes
- **Permanent Ban**: Potential ban from future ARC Prize competitions

### Appeals Process
- **Contact**: Email team@arcprize.org for appeals
- **Review**: Competition organizers will review appeals
- **Final Decision**: Organizer decisions are final

## Important Notes and Clarifications

### Competition Changes
- Rules may be updated for clarification during competition
- Major changes will be announced with advance notice
- Participants responsible for staying informed of rule updates

### Legal Considerations
- Participants must comply with all applicable laws
- Tax obligations are responsibility of prize winners
- Standard Kaggle terms of service apply

### Technical Limitations
- Solutions must work within Kaggle environment constraints
- No guarantee of hardware availability beyond stated minimums
- Backup plans recommended for technical issues

## Contact and Support

For questions about rules, technical issues, or clarifications:

- **Email**: team@arcprize.org
- **Discord**: https://discord.gg/9b77dPAmcA  
- **Twitter**: https://twitter.com/arcprize
- **Website**: https://arcprize.org/

**REMEMBER**: The core requirement is achieving 85% accuracy while adhering to all technical constraints and open source requirements. Success requires breakthrough advances in artificial general intelligence.