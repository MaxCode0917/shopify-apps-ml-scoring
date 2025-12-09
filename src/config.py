"""Configuration and constants for ISV Partner Propensity Model."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Shopify App Store Data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
OPENAI_BATCH_MODEL = "gpt-4o"  # For batch API

# Sampling Configuration
SAMPLE_SIZE = 2500
RANDOM_STATE = 42

# LLM Scoring Configuration
MAX_TOKENS = 500
TEMPERATURE = 0.3
BATCH_SIZE = 50  # For progress tracking

# Tier Thresholds (percentile-based since you can contact all 12K)
TIER_1_PERCENTILE = 90  # Top 10%
TIER_2_PERCENTILE = 60  # Next 30%

# Feature Engineering
INTEGRATION_KEYWORDS = [
    'api', 'integrate', 'integration', 'webhook', 'connect', 'sync',
    'automation', 'automate', 'workflow', 'zapier', 'rest', 'graphql'
]

SUPPORT_KEYWORDS = [
    'support', 'help desk', 'helpdesk', 'customer service', 'ticket',
    'chat', 'chatbot', 'faq', 'knowledge base', 'live chat', 'messaging'
]

ENTERPRISE_KEYWORDS = [
    'enterprise', 'business', 'professional', 'advanced', 'pro', 'plus',
    'premium', 'unlimited', 'custom', 'white label', 'whitelabel', 'b2b'
]

RAG_RELEVANT_KEYWORDS = [
    'search', 'product search', 'content', 'catalog', 'documentation',
    'knowledge', 'information', 'data', 'ai', 'artificial intelligence',
    'machine learning', 'ml', 'nlp', 'recommendation', 'personalization'
]

# High-value categories for CustomGPT.ai partnership
HIGH_VALUE_CATEGORIES = [
    'customer support', 'help desk', 'live chat', 'chatbot', 'faq',
    'search', 'product search', 'content management', 'knowledge base',
    'documentation', 'product display', 'product information'
]

# CustomGPT.ai Partner Program Context
PARTNER_CONTEXT = """
## About CustomGPT.ai
CustomGPT.ai is the #1 no-code platform for creating custom AI agents. Key offerings:
- RAG-As-A-Service API for production-ready applications
- Support for 1,400+ file formats and website ingestion
- 100+ integrations including Shopify
- SOC2 Type 2 certified, GDPR compliant
- Handles 93% of support and research questions automatically

## What We Seek in ISV Partners
Partners who can integrate our RAG APIs to:
1. Add AI-powered search and Q&A to their apps
2. Automate customer support within their platform
3. Build knowledge bases from merchant data
4. Enhance product discovery and recommendations

## Ideal Partner Profile
- Technical capability to integrate APIs
- Customer support, search, or content-related app functionality
- Established apps with real merchant traction
- Mid-market to enterprise merchant focus (aligns with our $99-499+/mo pricing)
- Apps that handle substantial content/data (benefit most from RAG)
"""

# LLM Prompt Template
LLM_PROMPT_TEMPLATE = """You are an expert at evaluating software companies for B2B partnership potential.

## Context
We run an ISV (Independent Software Vendor) Partner Program for CustomGPT.ai.

{partner_context}

## App to Evaluate

**App Title**: {title}
**Developer**: {developer}
**Rating**: {rating}/5 ({reviews_count} reviews)
**Pricing**: {pricing_hint}
**Last Updated**: {lastmod}
**Categories**: {categories}

**Description**:
{description}

**Key Benefits**:
{key_benefits}

**Review Sentiment Summary**:
{sentiment_summary}

## Task

Analyze this Shopify app and its developer as a potential ISV partner for CustomGPT.ai's RAG APIs.

Consider:
1. Could this app benefit from RAG-powered AI features (search, Q&A, knowledge base)?
2. Does the developer appear technically capable of API integration?
3. Is this a serious business with resources for partnership activities?
4. Does the app serve merchants who would value AI-powered features?

Respond ONLY with valid JSON:
{{
  "partner_probability": <float 0.0-1.0>,
  "reasoning": "<2-3 sentences explaining your assessment>",
  "rag_use_cases": ["<use_case_1>", "<use_case_2>"],
  "inferred_company_maturity": "<hobbyist|small_team|established_business|agency|enterprise>",
  "inferred_technical_sophistication": <int 1-5>,
  "inferred_business_model": "<free_only|freemium|paid|enterprise|unclear>",
  "inferred_merchant_segment": "<smb|mid_market|enterprise|all|unclear>",
  "partnership_fit_signals": ["<signal1>", "<signal2>"],
  "partnership_risk_flags": ["<flag1>", "<flag2>"],
  "confidence": "<low|medium|high>"
}}"""
