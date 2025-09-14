# GPT-5 Release Update (August 2025)

## Release Information
- **Release Date**: August 8, 2025, 10am PT
- **Status**: Now default in ChatGPT, replacing GPT-4o, o3, o4-mini, GPT-4.1, GPT-4.5

## Architecture & Variants
- **Unified System**: Smart/fast model + deeper reasoning model + real-time router
- **API Models**: gpt-5, gpt-5-mini, gpt-5-nano
- **Pro Version**: GPT-5 Pro (replaces o3-pro) with extended reasoning

## Technical Specifications
- **Input Limit**: 272,000 tokens
- **Output Limit**: 128,000 tokens (includes invisible reasoning tokens)
- **Modalities**: Text + image input, text output only

## Pricing (Significant Improvement)
- **GPT-5**: $1.25 input / $10 output per million tokens
- **Comparison**: 50% cheaper input than GPT-4o, same output price
- **Subscriptions**: 
  - Free: 10 messages/5 hours, then GPT-5-mini
  - Plus ($20/month): Higher limits
  - Pro ($200/month): Unlimited GPT-5 Pro

## Performance Benchmarks

### **Mathematics**
- **AIME 2025**: 94.6% (without tools)
- **GPQA**: 88.4% (GPT-5 Pro, new SOTA)

### **Coding Performance**
- **SWE-bench Verified**: 74.9% (beats o3's 69.1%, GPT-4o's 30.8%)
- **Aider Polyglot**: 88%
- **Leadership**: Currently leads SWE-bench Verified

### **Other Capabilities**
- **MMMU**: 84.2% (multimodal understanding)
- **HealthBench Hard**: 46.2%
- **τ2-bench telecom**: 96.7%
- **Scale MultiChallenge**: 69.6%

## Key Improvements

### **Reduced Hallucinations**
- **45% fewer factual errors** vs GPT-4o on web-enabled prompts
- **80% fewer factual errors** vs o3 with thinking mode
- **6x fewer hallucinations** than o3 on LongFact/FActScore

### **Efficiency**
- **50-80% less thinking time** than o3 for same performance
- Real-time routing optimizes model selection

### **Expert-Level Performance**
- Comparable to or better than human experts in ~50% of cases
- Spans 40+ occupations (law, logistics, sales, engineering)
- Outperforms o3 and ChatGPT Agent

## Impact on ARC Prize 2025

### **Coding Superiority**
- **74.9% SWE-bench** beats all previous models significantly
- **88% Aider Polyglot** excellent for code generation
- **50% cheaper input** makes API usage more viable

### **Reasoning Excellence**
- **94.6% AIME** shows strong mathematical reasoning
- **Unified reasoning system** perfect for ARC's abstract tasks
- **Extended reasoning mode** (GPT-5 Pro) for complex problems

### **Updated Recommendations for ARC Prize**
1. **GPT-5** now top choice for program synthesis ($1.25/$10)
2. **GPT-5 Pro** for complex reasoning (unlimited with $200/month)
3. **GPT-5-mini** for high-volume tasks (pricing TBD)

### **Cost Impact**
- Previous GPT-4o: $2.50/$10 → GPT-5: $1.25/$10 (50% input savings)
- Makes GPT-5 competitive with Gemini 2.5 Flash
- Justifies API usage over local models for accuracy

## Competitive Position vs Other Models

### **Coding Leadership**
- **GPT-5**: 74.9% SWE-bench (NEW LEADER)
- **Claude 4 Opus**: 72.5% SWE-bench  
- **OpenAI o3**: 69.1% SWE-bench
- **Kimi K2**: 65.8% SWE-bench

### **Overall Assessment**
GPT-5 appears to be a genuine breakthrough, not just incremental improvement. The unified reasoning system, dramatic hallucination reduction, and coding leadership make it highly relevant for ARC Prize 2025.

## Strategy Update Required
Given GPT-5's performance and pricing, the ARC Prize strategy should be updated to consider GPT-5 as the primary API choice, potentially replacing Gemini 2.5 recommendations.