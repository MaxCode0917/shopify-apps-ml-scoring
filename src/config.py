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
## Critical Context: We Want PARTNERS, Not Competitors

CustomGPT.ai provides RAG API infrastructure. We are looking for apps that would:
- INTEGRATE our RAG API into their existing product
- USE our backend to power AI features they don't have yet
- ENHANCE their product with our technology

We are NOT looking for:
- Apps that already have full AI chatbot/RAG solutions (they are COMPETITORS)
- Apps building their own AI infrastructure
- Apps that would compete with us rather than partner

## About CustomGPT.ai (What Partners Would Integrate)

CustomGPT.ai is an Enterprise RAG API platform. Partners integrate our API to add:

### What We Provide (Backend Infrastructure)
- **RAG-Powered Q&A Engine**: Partners send questions, we return accurate answers from their knowledge base
- **Knowledge Base Indexing**: We ingest their data (docs, websites, products) and make it searchable
- **Citation-Backed Responses**: >95% accuracy, no hallucinations, source attribution
- **Streaming Responses**: Sub-second real-time answers
- **Multi-Language**: 90+ languages supported
- **Enterprise Security**: SOC-2 Type II certified

### How Partners Would Use Us
- Add "Ask AI" feature to their existing app
- Power a customer support chatbot within their platform
- Enable conversational product discovery
- Build FAQ automation without building RAG infrastructure
- Add AI-powered search to complement their existing features

## Ideal Partner Profile (INTEGRATE our API)

EXCELLENT PARTNERS (score 0.7-0.9):
Apps that have a clear use case for RAG but DON'T have it yet:

1. **Help Center/FAQ Apps WITHOUT AI**: Build FAQ pages manually - could add AI Q&A powered by us
2. **Product Review Apps**: Have review data but no AI analysis - could add review Q&A chatbot
3. **Search & Filter Apps WITHOUT AI**: Keyword-based search - could add semantic/conversational search
4. **Customer Support/Helpdesk Apps**: Ticketing systems without AI responses - could automate with our RAG
5. **Knowledge Base Builders**: Static docs - could add intelligent Q&A layer
6. **Size Guide/Fitting Apps**: Static charts - could add conversational size recommendations

GOOD PARTNERS (score 0.5-0.7):
Apps that could add AI features to enhance their core offering:

1. **Loyalty/Rewards Apps**: Could add chatbot for program questions
2. **Subscription Apps**: Could add AI support for subscription management queries
3. **Translation Apps**: Could enhance with AI-powered multilingual Q&A
4. **Forms/Survey Apps**: Could add intelligent form assistance

POOR PARTNERS (score 0.1-0.3):

1. **AI Chatbot Apps (COMPETITORS)**: Already have AI chatbots - they compete with us, not partner
   - Examples: Apps with "AI Chatbot", "ChatGPT-powered", "AI Assistant" in name/description
   - They build their own AI, won't use our API

2. **SEO/Google Optimization Apps**: Wrong domain - optimizing for Google rankings, not site Q&A

3. **Invoice/Document Generation**: Template-based output, no knowledge retrieval needed

4. **Pixel Tracking/Analytics**: Backend data collection, no customer-facing AI opportunity

5. **Data Feed/Export Apps**: Structured data transformation, not conversational

6. **Countdown Timer/Popup/Banner Apps**: Simple display widgets, no AI opportunity

7. **Shipping Label/Logistics Apps**: Operational tools, not knowledge-based

## Key Distinction: Partner vs Competitor

ASK: "Would this app BUY our API or BUILD their own?"

PARTNER signals:
- No current AI features (gap we can fill)
- Core product is NOT AI (AI would be add-on feature)
- Technical capability to integrate APIs
- Customer base that would value AI features
- Pain point that RAG solves (support tickets, repetitive questions)

COMPETITOR signals:
- Already has AI chatbot or AI-powered features
- "AI" or "ChatGPT" is core to their value proposition
- Building their own ML/AI infrastructure
- Would view us as threat, not enabler
"""

# LLM Prompt Template
LLM_PROMPT_TEMPLATE = """You are evaluating Shopify apps for PARTNERSHIP potential with CustomGPT.ai.

{partner_context}

## Evaluation Criteria

1. **Partnership Fit (50%)**:
   - Do they have a CLEAR USE CASE for RAG that they don't have yet?
   - Would AI be an ADD-ON feature (not their core product)?
   - Is there a gap we can fill?

2. **Integration Readiness (25%)**:
   - Do they have API integrations already? (shows capability)
   - Multiple apps? (development resources)
   - Enterprise pricing tier? (budget for API costs)

3. **Business Viability (15%)**:
   - Strong ratings/reviews (healthy business)
   - Paid plans (customers willing to pay)
   - Growth potential

4. **Competitive Risk (10% - NEGATIVE factor)**:
   - SUBTRACT points if they have existing AI features
   - SUBTRACT points if AI is their core differentiator
   - SUBTRACT points if they're likely building in-house AI

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

## Output Requirements

partner_probability: 0.0-1.0
- 0.7-0.9: Excellent partner - clear RAG use case, no competing AI features
- 0.5-0.7: Good partner - could benefit from RAG, moderate fit
- 0.3-0.5: Weak partner - limited use case or some competitive overlap
- 0.1-0.3: Poor partner - competitor, wrong domain, or no clear use case

IMPORTANT:
- Apps with "AI Chatbot" as core feature = LOW score (competitor)
- Apps with "Search" for SEO = LOW score (wrong domain)
- Apps with NO AI but clear Q&A/support need = HIGH score (ideal partner)
- High developer sophistication + no AI features = EXCELLENT partner opportunity

Respond ONLY with valid JSON:
{{
  "partner_probability": <float 0.0-1.0>,
  "reasoning": "<2-3 sentences: Is this a PARTNER (would buy our API) or COMPETITOR (has AI already)? What RAG use case exists?>",
  "is_competitor": <boolean - true if they already have AI chatbot/search features>,
  "existing_ai_features": ["<list any AI features they already have, or empty if none>"],
  "rag_use_cases": ["<1-3 specific ways they could integrate CustomGPT.ai API, or empty if competitor>"],
  "inferred_company_maturity": "<hobbyist|small_team|established_business|agency|enterprise>",
  "inferred_technical_sophistication": <int 1-5>,
  "inferred_business_model": "<free_only|freemium|paid|enterprise|unclear>",
  "inferred_merchant_segment": "<smb|mid_market|enterprise|all|unclear>",
  "partnership_fit_signals": ["<signal1>", "<signal2>"],
  "partnership_risk_flags": ["<flag1>", "<flag2>"],
  "confidence": "<low|medium|high>"
}}"""




