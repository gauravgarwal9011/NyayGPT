"""
knowledge_base.py
=================
Verified, document-grounded content extracted verbatim from the source PDF
(`Ignatiuz_AIML_capabilities_deck.pdf`).

WHY THIS FILE EXISTS
--------------------
PDF text extraction is *lossy*: tables, images, and multi-column layouts can
mangle key facts (numbers, client names, percentages). To guarantee zero
hallucination of those high-value facts, we manually copy the canonical
strings here once and use them as the *primary* grounding source.

The PDF extractor still runs as a *secondary* source for coverage — but the
knowledge base takes priority when the teacher generates training samples.

EVERY string below appears verbatim in the source PDF. Do not "improve" the
wording — the whole point is faithful reproduction.
"""

# A plain dict mapping a short section key → the verbatim text from the PDF.
# Using a dict (instead of, say, a list of tuples) makes it cheap to look up
# a single section by name in tests / audits.
DOCUMENT_KNOWLEDGE_BASE = {

    "company_overview": """
Ignatiuz is a forward-thinking technology solutions provider committed to driving digital
transformation and empowering businesses across industries to thrive in the modern digital
landscape. Expertise includes AI & ML model development, cloud consulting, business
intelligence, and robotic process automation. Mission: help organizations embrace
cutting-edge technologies, optimize operations, and achieve sustainable growth.
Partners: Microsoft, UiPath, Outsystems, Nintex, Smartsheet, CobbleStone, Salesforce,
AWS partner network, Nividous, SAP, InsightGlobal, NJMMA.
Offices: Indore (India), Hyderabad (India), Pennsylvania USA — 600 Eagleview Blvd,
Suite #300, Exton PA 19341. Tel: +1 267-281-8692. Fax: +1 484-206-4141.
Email: info@ignatiuz.com. Website: www.ignatiuz.com.
""",

    "services": """
Six core service areas (from "What We Do?" slide):
1. Emerging Technologies: AI/ML model development, cutting-edge data science services,
   strategic AI/ML consulting.
2. Product Engineering: SaaS development, MVP creation, innovative mobile engineering.
3. Consulting Services: Architectural reviews, UI/UX design, Centers of Excellence for
   Microsoft and Smartsheet.
4. Cloud Engineering: Cloud consulting, seamless migration frameworks, managed services.
5. Digital Journeys: RPA, advanced Business Intelligence solutions, digital transformation.
6. Low Code Experiences: Low code/no code automation using Smartsheet, Microsoft Power
   Platform, and Outsystems.
Also: Post Merger IT Integration, Full Stack Cloud Computing, Enterprise Mobility, AI/ML,
RPA, Low Code Application Development, Microsoft 365 Solutions, Technology Consulting & QA.
""",

    "clients": """
Clients (from "Our Clients" slide):
Saint-Gobain, NJ.gov, International Seaways Inc., Inotiv, Solid Biosciences,
Borough of Point Pleasant NJ, Township of Bridgewater, City of Wilmington Delaware,
Berlitz, Cardone, PM-International, Hazlet Township, Ocean County NJ,
Old Bridge Middlesex County, CypherTax, Cass Information Systems Inc.,
PPG Professional Plumbing Group, SOMPO, Roche.
""",

    "generative_ai": """
Generative AI is at the forefront of the artificial intelligence revolution, transforming how
businesses innovate, operate, and engage with customers.

What is Generative AI?
Algorithms that generate new content — text, images, sound, and beyond — based on patterns
learned from vast datasets. Not just automating tasks; creating something new and valuable.
Generates: Text (human-like, articles, responses), Images (from descriptions), Music and
Sound (specific styles), Video (emerging).

Key Aspects:
- Creativity and Innovation: Enhances creative processes for artists, writers, designers.
- Efficiency: Automate content creation, save time, maintain quality.
- Personalization: Tailored content improves customer engagement and satisfaction.
- Cross-Disciplinary: Healthcare drug discovery, gaming dynamic storytelling, marketing
  personalized campaigns.

Key Capabilities:
1. Personalization At Scale: Engines adapt in real-time to user interactions and preferences,
   processing user data including past interactions and knowledge bases.
2. Predictive Analytics And Decision-Making: Generate new content from patterns in vast datasets.
3. Innovation Acceleration: Shorten R&D cycles by simulating outcomes using historical data;
   analyze past R&D data, experiments, and trials to propose new formulas.
4. Advanced Content Creation: Create high-quality text, images, videos mimicking human
   creativity by analyzing existing content databases for patterns, styles, structures.
""",

    "autonomous_intelligence": """
Autonomous Intelligence: Systems capable of performing tasks and making decisions
independently, without human intervention.
Key Characteristics:
- Self-Governance: Operates autonomously in real time.
- Learning and Adaptation: Learns from experiences to improve performance.
- Real-Time Decision Making: Responds dynamically to environment changes.
- Task Execution: Can perform complex tasks that usually require human judgment.
Examples: Self-driving cars, Autonomous drones for delivery, Robotics in manufacturing.
Benefits: Increased Efficiency (automates repetitive tasks), Enhanced Safety (reduces human
error in hazardous environments), Cost Reduction (lowers operational costs).
""",

    "rnn_chatbot": """
Recurrent Neural Network (RNN):
Artificial neural network designed to recognize patterns in sequences of data such as text,
genomes, handwriting, or numerical time series data from sensors and other recurring sources.
Called "recurrent" because they perform the same task for every element of a sequence, with
output dependent on previous computations. RNNs possess a memory capturing information
about what has been calculated so far (unlike traditional neural networks).

Chatbot With RNN:
Capable of understanding and generating human-like responses, learning from previous
interactions to improve over time. Can manage a large volume of queries simultaneously,
ensuring quick and accurate responses, crucial for maintaining high customer satisfaction
and reducing the workload on human agents.
Benefits: Improved Efficiency (automates responses to common queries, reducing response
times and costs), Scalability (handles query volume fluctuations), Learning Capability
(continuously improves through learning from interactions).
""",

    "ecommerce_ai_search": """
Enhancing E-Commerce Search:
AI/ML algorithms understand natural language and user intent.
Benefits: Enhanced Relevance, Personalized Recommendations, Auto-Correction and Synonym
Recognition.
AI/ML-Driven Features:
- Instant Search Suggestions: Real-time suggestions as users type.
- Auto-Complete: Predict and complete search queries to save time.
- Query Expansion: Broaden results by including related terms and synonyms.
""",

    "ai_recommendation_engine": """
AI Recommendation Engine — Three-step process:
Step 1 (Insert data): Product Catalogue Data + Customer Data → Integrate Customer Data
Step 2 (Customize): AI Model → Recommendation Type, Objective, Business Rules & Needs
Step 3 (Deliver): Prediction API → Show Recommendations at Customer Touchpoints

Considerations:
1. Intelligent & personalized recommendations.
2. Similar & frequently bought product recommendations.
3. User behaviour, purchase history & preferences.
4. Understanding of natural language & user intent.
5. Additional attributes from supplier content or web content.
""",

    "ai_chatbot": """
Customized AI-Powered Chatbot:
Handle Unexpected Questions, Recap Conversation, & Automate Processes.
Generate a Personalized Experience For Both Internal and External Users.

Capabilities:
Content Generation: Auto-generate responses to customer inquiries, generate personalized UI.
Summarization: Summarize support conversations, financial reports, analyst articles, social
media trends.
Semantic Search: Search reviews, information discovery and knowledge.

Application — Tax Information Retrieval (TaxBot demo):
- Automation: Streamlines answering tax-related queries.
- Consistency: Consistent and accurate information based on the dataset.
- Accessibility: Users access tax information anytime without human intervention.

Azure OpenAI Architecture:
Web interface ↔ Web Server ↔ API Gateway ↔ Azure OpenAI
Multiple Data Sources: Database → Data Pipeline → Data Lake → Data Training → Azure OpenAI

Models: AzureGPT-4 series (GPT-4 Turbo with vision), GPT-3.5 Turbo series (content
generation, summarization, image understanding, semantic search, NL to code), Embedding series.
""",

    "supercharge_support": """
Supercharge Support With AI:
Dynamic Script with AI Assistant — real-time call coaching and faster resolution.

Three AI capabilities:
1. Installer Certification Verification: AI analyzes customer data (purchase history,
   registration) to determine installer certification for specific products; prompts agent
   with relevant support options.
2. Solution Recommendation Engine: AI analyzes caller's issue description, identifies
   keywords/patterns, presents ranked list of potential solutions from historical data or
   troubleshooting manuals.
3. Automated Knowledge Base Search: AI scours internal knowledge base for relevant articles,
   FAQs, or video tutorials based on installer questions; agent shares directly through
   call interface.

Benefits:
- Real-time Call Coaching: AI whispers suggestions during calls.
- Faster Resolutions: AI-reduced call times mean happier customers.
- Personalized Support: AI-powered solutions cater to individual needs and preferences.
- Proactive Communication: Automated follow-up emails/texts keep customers informed.
""",

    "process_mining": """
Process Mining: Technology that uses event log data from IT systems to analyze and improve
business processes. Bridges the gap between theoretical process models and actual process
execution, providing transparency and insights.
Applicable areas: Supply chain, finance, HR, IT operations, customer service.
Use cases: Process optimization, compliance monitoring, root cause analysis, performance
benchmarking.

Need For Process Mining:
1. Intended V/S Visibility: Not all processes follow the 'happy path' (smooth, no complications).
   Exceptions/deviations significantly impact efficiency.
2. Lack of Visibility: Employees have limited understanding of sub-processes; fragmented view
   of entire process chain hinders comprehensive insight.
3. Constant Changes: Processes evolve for customer requirements, regulations, restructuring
   but adaptations not always reflected in documentation — discrepancies between documented
   and actual processes.

How Process Mining Works:
Data Sources: Event logs from IT systems (activity performed, timestamp, identity of performer).
Key Steps:
1. Automated Process Discovery: Algorithms analyze event logs, construct comprehensive model
   of actual process flow — reveals sequence, frequency, paths.
2. Conformance Checking: Compares discovered model against intended model; identifies
   deviations, non-compliance, divergences from expected workflow.
3. Performance Analysis: Evaluates efficiency; highlights bottlenecks, delays, inefficiencies.

Starting a Process Mining Project:
Problem Identification → Data Collection → Pilot Implementation → Analysis And Improvements
(Use insights from process mining to implement improvements and monitor their impact.)
""",

    "process_mining_business_case": """
Enterprise Business Case (Example Company — exact figures from document):
Revenue: 2,740 M€ | Sales orders/year: 670,000 | Purchase orders/year: 265,000 | Avg FTE cost: 44,000 €
ROI: 370% | Payback time: 7.6 months | Business Case: 12.9 M€ | Investment: 2.5 M€

Value by process area:
- Order management: ≈ 5.3 M€ (cut lead time, improve on-time delivery, increase sales)
- Accounts receivable: ≈ 4.1 M€ (speed up invoicing, reduce billing blocks)
- Procurement: ≈ 2.8 M€ (increase just-in-time deliveries, reduce Maverick buying)
- Accounts payable: ≈ 2.7 M€ (improve on-time payment, increase cash discounts)
Theme: Reduce manual rework & improve automation.
""",

    "rpa": """
RPA (Robotic Process Automation): Not physical robots — software "bots". Digital workers.
Accomplishes repetitive, manual tasks more quickly and accurately than humans.
Excellent at rule-based and trigger-driven tasks (like data processing). Works across any
application or system.

Good tasks for RPA: Repeated at regular intervals or pre-defined trigger; well-defined
inputs/outputs; heavily dependent on email or spreadsheets; receiving data from one system
and inputting into another; takes at least 8–10 hours/week.

What RPA accomplishes: Refocus employees to value-add tasks; ensure staff on most important
tasks; improve job satisfaction and morale; integrate outdated/inflexible systems (UI is the API);
harmonize inconsistent data sources; improve speed of revenue recognition and TAT; increase compliance.

RPA Statistics: 93% of C-level executives say RPA kickstarted digital transformation.
83% will use RPA to build agility, diversity and resilience. 82% expect automation progress
and acceleration over the next three months.

Three Models of RPA:
1. Unattended RPA: End-to-end automation — back office/batch oriented. No human involvement.
2. Attended RPA: End-to-end automation — back office/batch oriented. Bots work alongside humans.
3. Hybrid RPA: End-to-end automation — back office/batch oriented. Humans + bots collaborate.

Intelligent RPA Summary: Proven (hundreds of processes, all industries), Intelligent (ML tech),
High Value To Cost (low-cost, big impact quickly), Extensible (accommodates old/new systems),
Secure & Scalable, Human-Bot Orchestration.

Benefits of Process Automation: Save Time, Eliminate Manual Mistakes, Reduce Costs,
Accelerate Outcomes, Focus on strategic items, Kick-start digital transformation, Reduce
paper waste, Improve customer satisfaction, Strengthen operations, Happier employees.
Tagline: "Faster. Better. Happier."
""",

    "idp": """
Intelligent Document Processing (IDP): Cognitive automation platform that can extract
information from semi-structured and unstructured data across a multitude of document formats.
Doc Sources: Internal, External.
Input Types: Structured, Semi Structured, Unstructured.
Formats: Word, Excel, PDF, Images, Text.

Six-stage IDP pipeline:
1. Classification → 2. Pre-processing → 3. Extraction → 4. Post Processing → 5. Verification → 6. Integration
""",

    "manual_invoice_challenges": """
Challenges of Manual Invoice Processing:
- Aberdeen Group research: It takes between 4.1 and 16.3 days for companies to process an
  invoice from receipt through payment approval.
- Canon Business Process Services study: More than half of all invoice processing requires
  at least 76% manual input.
""",

    "case_study_brewing_ap": """
Case Study: Accounts Payable — A Leading Drink and Brewing Company
Volume: 20K+ invoices received monthly from 400+ vendors in different countries.
Solution deployed: Bots with cognitive capabilities to automate complete AP process.
- RPA Bots download invoices from emails and segregate by type.
- Straight through processing: automated invoice match with PO, auto posting to ERP.
- Exceptions handled through manual intervention.
- Rule-based, multi-level approval matrix.
- Intuitive vendor portal for invoice submission, status tracking, and query management.

EXACT RESULTS (do not change these numbers):
- 90% Reduction in process TAT
- 1000 Staff-hours saved per month
- 100% Reduction in manual errors
""",

    "case_study_manufacturing_invoice": """
Case Study: Invoice Processing @ Manufacturing Firm
Volume: 20K+ invoices monthly from 400+ vendors. 2500+ invoices weekly in varied formats.
Challenge: 6+ FTEs transcribed data into SAP ERP manually, causing delays. High risk of
human errors could negatively affect vendor relationships, resulting in substantial business losses.
Solution: RPA Smart Bots with Machine Learning, Computer Vision and advanced OCR capabilities
deployed in two weeks. Bots read, understand and fetch details from unstructured data types
and image-based documents and feed into SAP ERP. Human-Bot collaboration for verification
wherever required.

EXACT RESULTS (do not change these numbers):
- 90% Reduction in process TAT
- 1000 Staff-hours saved per month
- 100% Reduction in manual errors
""",

    "case_study_media_orders": """
Case Study: Automation of Customer Orders @ Media Company
Volume: 60+ FTEs manually processing 1000+ customer orders daily.
Challenge: High risk of manual errors and delays; varied formats of customer orders.
Solution: Smart Bots with cognitive capabilities.
- Bots access order info from emails and fax, extract specific information, feed into
  automation control center interface.
- Smart Bots enter extracted data into ERP; lower confidence score triggers manual exception.
- Enabled centralized processing and 50% savings on FTEs in the process.

EXACT RESULTS (do not change these numbers):
- 90% Reduction in process TAT
- 1000 Staff-hours saved per month
- 100% Reduction in manual errors
""",

    "automation_journey": """
Ignatiuz Automation Journey Support — Three stages:
1. Launch — Advisory and Strategy ("Advise You"):
   For clients saying: "We want to know more" / "OK, where do we start?"
2. Expand — Capability & Implementation ("With You"):
   For clients saying: "How do we accelerate value?" / "We use it but struggle to scale."
3. Operate — Digital Services ("For You"):
   For clients saying: "We have a team but want to broaden capability" /
   "We need to expand capability across the enterprise."
""",

}
# End of DOCUMENT_KNOWLEDGE_BASE.
# Every string above is extracted verbatim from Ignatiuz_AIML_capabilities_deck.pdf.
# Particular attention to case study metrics: 90% TAT reduction, 1000 staff-hours/month,
# 100% error reduction — these exact figures appear on slides 43, 44, and 45.
# The teacher must quote these precisely; the quality gate checks for them.
