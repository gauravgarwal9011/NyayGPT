# Job Search Session — Context Document (continuation of NyayaGPT/chat_context.md)

**Date:** 2026-05-02
**Scope:** Resume rebuild + LinkedIn/Medium/HF promotional materials + application strategy
**User:** Gaurav Garwal (gauravgarwal9011@gmail.com)
**Working dir:** `/home/ubuntu/Fine-tuning/`
**Artifacts dir:** `/home/ubuntu/Fine-tuning/job_search/`

---

## TL;DR

User asked: "If I had to start all over again, how should I build my resume to get maximum interview calls, and whom should I connect with first?" Session produced four execution artifacts in `job_search/` and a posting playbook for the NyayaGPT blog. Initial drafts had fabrications; user pointed to `NyayaGPT/chat_context.md` as canonical truth and all artifacts were rewritten against it. HF adapter went live mid-session and links were added to every doc. PDF compile deferred to Overleaf at user's request.

---

## Canonical Identity & Links (use these everywhere)

- **GitHub:** `https://github.com/gauravgarwal9011/NyayGPT` ← **typo'd repo name** (`NyayGPT`, not `NyayaGPT`); not yet renamed
- **HF Adapter (LIVE):** `https://huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter` ← username is `gauravgarwal`, NOT `gauravgarwal9011` (different from GitHub handle)
- **LinkedIn:** `https://www.linkedin.com/in/gauravgarwal/`
- **Email:** `gauravgarwal9011@gmail.com`
- **Phone:** `+91 88712 19166`
- **Location:** Indore, India

---

## Strategic Advice Given (Resume + Job Search)

### Resume restructuring decisions
- Title: **AI/ML Engineer (LLMs, RAG & Agentic Systems)** — not "Data Scientist". Aligned across header, summary, work experience.
- Order: Header → 3-line Summary → Core Skills → **Featured Projects → Work Experience** → Education. Projects before experience because work tenure is short (~11 months) but project signal is mid-level.
- Skills: trimmed from 7 lines to 5. Removed junior-NLP signals (POS tagging, NER as standalone) and any tool not actually used.
- Design Engineer (2020–2023): compressed to one sentence, framed as "pre–data-science career".
- Removed Maruti Suzuki internship (per existing chat_context.md decision).
- Quantification: keep specific reproducible numbers (3.3× memory, 2.2× faster, ROUGE 0.57); cut vague %s (35% inference speed, 22% accuracy, 25% retention) where they read as filler.

### Application strategy (priority order)
1. **Referrals (60% of effort)** — personalized LinkedIn DMs to engineers who joined target companies in the last 6 months. ~15–25% conversion.
2. **YC / seed-stage AI startup careers (20%)** — wellfound.com, ycombinator.com/jobs, workatastartup.com. Filter "LLM" / "GenAI" + India/remote.
3. **Direct DMs to founders/eng-leads of Indian GenAI startups (15%)** — Sarvam AI, Krutrim, AI4Bharat, CoRover, Yellow.ai, Haptik, Observe.AI, Fractal, Quantiphi.
4. **Career portals (5%)** — LinkedIn Easy Apply / Naukri as backstop only; ~1% conversion.

### Networking priority
1. AI/ML Engineers at target companies who joined in last 6 months
2. Engineering Managers / Hiring Managers on GenAI teams
3. Indian AI community leaders (HF / AI4Bharat / Sarvam)
4. AI-focused recruitment agencies (Uplers, Flexiple, Turing, Toptal)
5. IIT Roorkee + Intellipaat cohort

### Salary band positioning (Indian market, 2026)
- Indian GenAI startups: ₹18–32 LPA fixed
- Indian services with GenAI focus (Fractal, Quantiphi, Mu Sigma): ₹22–40 LPA
- Top tier (Sarvam senior bands, MAANG India AI teams): ₹35–60+ LPA — but MAANG India requires DSA prep that resume doesn't currently signal

### 90-day plan
- Weeks 1–2: Resume rebuild + LinkedIn headline + HF Space for NyayaGPT
- Weeks 3–4: NyayaGPT blog post + Twitter thread + open-source one component
- Weeks 5–8: 5 personalized referral DMs/day, M–F
- Weeks 9–12: Interview, leverage early offers
- Realistic outcome with discipline: 15–25 first-rounds, 4–8 onsites, 2–3 offers

---

## Files Created in This Session

All in `/home/ubuntu/Fine-tuning/job_search/`:

### `resume_v2.tex`
- Single-page LaTeX (Charter font, 0.45in margins, parskip 1pt).
- Structure: Header (with GitHub + LinkedIn + HuggingFace links) → Summary (3 lines) → Core Skills (5 lines) → Featured Projects (NyayaGPT, Due Diligence, English Speaking, Multimodal RAG, Route Guidance) → Work Experience (Ignatiuz, Design Engineer one-liner) → Education.
- NyayaGPT bullets (corrected against canonical NyayaGPT/chat_context.md):
  - Mistral-7B-Instruct-v0.3, Unsloth + QLoRA, r=16, α=32, **2 epochs** (not 3), dropout=0.0, 1,690 pairs (1,521 train / 169 eval).
  - Eval: ROUGE-1 = 0.57, ROUGE-L = 0.40, RAGAS Faithfulness = 0.66.
  - Quantization: **three-tier** FP16 / INT8 / INT4 (Q4_K_M) GGUF — NOT four-tier (early draft had a fabricated Q4_0 row).
  - Headline: 14.5GB → 4.4GB (3.3×), 10.8 → 4.9 ms/tok (2.2×), zero ROUGE-L drop.
  - **Blackwell cuBLAS bullet** added — explicitly calls out diagnosing the CUDA 12.8 / sm_120 regression and pivoting all inference to llama.cpp. This is the strongest differentiator on the resume.
  - Day 5 framing: "vanilla Mistral Q4 vs NyayaGPT Q4 apples-to-apples" — NOT FP16-vs-INT4.
- User will compile via Overleaf — local pdflatex/tectonic install was offered (~50 MB tectonic, ~800 MB texlive minimal) and declined.

### `nyayagpt_blog_post.md`
- ~1,500 words, Medium/HF format with frontmatter.
- **Hook is the Blackwell cuBLAS catastrophe**, not "I quantized a model". The opening makes the cuBLAS-broke / llama.cpp-saved-me story the spine of the post.
- Sections: TL;DR with table → Why Indian legal → Dataset → QLoRA config (3 non-default decisions) → cuBLAS catastrophe (4 attempts documented) → Quantization benchmark (real numbers from `output/benchmark_results.json`) → Day 5 apples-to-apples A/B → MLOps stack → What I'd do differently → What's next → Closing advice.
- Real benchmark numbers used (from canonical doc): FP16 14.50 GB / 10.8 ms / 0.367 ROUGE-L; INT8 7.70 GB / 6.8 ms / 0.371; Q4_K_M 4.37 GB / 4.9 ms / 0.371. Memory column noted as "GGUF size on disk" (nvidia-smi unavailable in WSL2).
- No fabrications: removed claims of "200-prompt eval set" (real: 169), "2.5 hours on A100" (real: RTX 5090, time not stated), "14 hyperparameter sweeps" (made up), "BNS coverage" (not in dataset), per-variant RAGAS-F (only ROUGE-L per variant in real JSON).
- HF adapter URL embedded in TL;DR + MLOps stack + closing.

### `referral_dm_templates.md`
- 5 templates: Indian GenAI Startup / MAANG India / Indian Services / YC AI Startup / Recruiter.
- Plus universal rules (Tue–Thu 9–11am or 4–6pm IST, <120 words, never ask for job in first message, lead with project link, follow up once after 7 days).
- Plus follow-up template + tracking instructions.
- Canonical-links block at top includes both GitHub (with NyayGPT typo flag) and live HF adapter URL.
- Template 5 (recruiter) leads TL;DR with HF Hub URL — recruiters find HF artifacts especially credible.

### `target_companies.csv`
- 33 companies, columns: Priority, Company, Tier, Type, Location, Target_Role, Why_Fit, Hiring_Contact, Channel, JD_Link, Date_Applied, Status, Last_Activity, Notes.
- Header has 3 comment lines listing canonical artifacts (GitHub, HF Hub, LinkedIn) for paste-into-DM convenience.
- Priority 1 (focus first 30 days): Sarvam AI, Krutrim, AI4Bharat, Google DeepMind India, Microsoft IDC.
- Priority 2 (~15 companies): Adobe MDSR, Nvidia India, Anthropic, Cohere, Hugging Face, Razorpay, Flipkart, Swiggy, Zomato, Meesho, PhonePe, Together AI, Replicate, Modal Labs.
- Priority 3 (~13 companies): Indian services tier, conversational AI, AI products, recruiters.

---

## Critical Corrections Made (initial drafts → corrected)

User flagged "the resume v2 and blog post are wrong" and pointed to `NyayaGPT/chat_context.md`. Errors fixed against canonical truth:

| What was wrong | Corrected to |
|---|---|
| Repo URL `github.com/gauravgarwal9011/NyayaGPT` | `github.com/gauravgarwal9011/NyayGPT` (typo'd repo name) |
| 3 epochs training | 2 epochs (`adapters-2ep` won over `-3ep`) |
| LoRA dropout 0.05 | 0.0 |
| Hardware "A100 40GB" | RTX 5090 (Blackwell sm_120), WSL2, CUDA 12.8 |
| 4-tier benchmark (FP16/INT8/Q4_K_M/Q4_0) | 3-tier (FP16/INT8/INT4 Q4_K_M) — Q4_0 row was fabricated |
| Per-variant RAGAS-F column | ROUGE-L only (real JSON has 3 metrics: Memory, Latency, ROUGE-L) |
| 200-prompt eval set | 169 (1,521/169 train/eval split) |
| Day 5 = "FP16 vs INT4 NyayaGPT" | Day 5 = "vanilla Mistral Q4 vs NyayaGPT Q4" (apples-to-apples) |
| Tok/s figures (92.6, 144.9, 204.1) | Removed (fabricated) |
| "Memory" labeled as VRAM | Labeled as GGUF on-disk size (nvidia-smi unavailable in WSL) |
| HF Hub link `huggingface.co/gauravgarwal9011` (in progress) | `huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter` (LIVE) — note different username |

---

## Medium + Hugging Face Posting Playbook (delivered to user)

### Canonical surface decision
**Recommended: HF Community Blog as canonical, Medium as syndicated mirror.** Reasons: HF organic ML traffic, persistence, sits next to live adapter. Medium gets `canonical_url` pointing to HF.

### HF Community Blog flow
1. `gh repo fork huggingface/blog --clone`
2. Add post as `nyayagpt-blackwell-cublas.md` with HF frontmatter format
3. Drop 1200×630 thumbnail at `blog/assets/nyayagpt/thumbnail.png`
4. Edit `_blog.yml` index
5. PR to `huggingface/blog`, wait 1–7 days for review
6. Independently update model card README on the live adapter repo

### HF Posts (short-form, after blog goes live)
- `huggingface.co/new-post`
- Lead with Blackwell cuBLAS hook, table, three artifact links
- Tag `@mistralai`, `@unsloth`

### Medium flow
- Option A (clean): import-from-URL once HF is live; auto-sets canonical
- Option B: paste markdown directly, manually set Canonical URL in Advanced Settings
- Title + subtitle preserved; tags = `Machine Learning`, `Large Language Models`, `Fine Tuning`, `Quantization`, `India`
- **Apply to publications** (Towards AI, Generative AI, Better Programming, Towards Data Science, Level Up Coding) — apply today, publish under best acceptance later

### 7-day promotion sequence
- Day 0 (publish): HF blog PR + model card update
- Day 0 evening: HF Post
- Day 1: Medium published with canonical URL → HF; LinkedIn post → Medium
- Day 2: X/Twitter thread (8–10 tweets) tagging `@unsloth @MistralAI @llamacpp_ai`
- Day 3: r/LocalLLaMA + r/MachineLearning [P] — lead with technical hook, NOT self-promo
- Day 5: reply to every comment
- Day 7: review traffic stats

### Don'ts flagged
- Don't publish without canonical URL (SEO penalty)
- Don't post all platforms same day (stagger 24h)
- Don't lead Reddit with self-promo
- Don't use AI-generated thumbnails (segment can spot them)

---

## Outstanding from Previous Session (NyayaGPT/chat_context.md)

Carried forward — still pending:
1. **URGENT: Revoke leaked HF token** (`hf_BRSouBZh...` was hardcoded in `06_hf_hub_deployment.ipynb` cell-3, pushed to GitHub in commit `275df9e6`, blocked by GH Push Protection but still compromised). Token was visible in GH error logs and chat transcript. **Action:** revoke at https://huggingface.co/settings/tokens.
2. **Delete cell-3 from `06_hf_hub_deployment.ipynb`** — user previously declined the proposed `NotebookEdit delete` operation. Verify gone before next push.
3. Day 7: Build `07_hf_spaces_demo.ipynb` (Gradio app, free CPU tier, INT4 GGUF).
4. (Optional) Rename GitHub repo `NyayGPT` → `NyayaGPT`. If done, update `resume_v2.tex` and `nyayagpt_blog_post.md` URLs.

**Note on item 1+2:** HF adapter went live this session (`huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter`), so the user did successfully push at some point — implying either token cleanup is done, or a new token was used. Should confirm before next chat assumes the token is rotated.

---

## Outstanding from This Session

Pending — pick up next chat:
1. **Compile resume to PDF via Overleaf** — verify single-page fit. User chose Overleaf over local install.
2. **Verify HF adapter README quality** — the public model card needs: working usage snippet, eval metrics table, link back to GitHub, public visibility. A 404 or empty card from a recruiter click is worse than no link.
3. **Open HF Community Blog PR** to huggingface/blog with the rewritten post.
4. **Apply to Medium publications** (Towards AI, Generative AI, Better Programming, TDS, Level Up Coding) — today.
5. **Send first 5 referral DMs** to Priority 1 companies in target_companies.csv.

Optional next-chat asks the user has open:
- Draft X/Twitter thread (8–10 tweets) for the post
- Draft Reddit r/LocalLLaMA post in correct technical-first tone
- Write the HF model card README content
- Personalized first lines for top 5 companies in tracker

---

## Useful Paths

```
/home/ubuntu/Fine-tuning/job_search/
├── resume_v2.tex                  # one-page LaTeX, Charter, ready for Overleaf
├── nyayagpt_blog_post.md          # ~1500 words, Medium/HF format, Blackwell cuBLAS hook
├── referral_dm_templates.md       # 5 templates + follow-up + universal rules
├── target_companies.csv           # 33 companies, P1/P2/P3 tiered
└── chat_context_2.md              # this document

/home/ubuntu/Fine-tuning/NyayaGPT/
├── chat_context.md                # canonical project truth — refer to this for any
│                                  #   NyayaGPT-specific number, decision, file path
├── output/benchmark_results.json  # real quantization numbers (3 variants)
└── ...                            # full project layout in chat_context.md
```

---

## Hand-off to next chat

Key facts to know cold:
1. **Canonical truth lives in `NyayaGPT/chat_context.md`** for any NyayaGPT-specific claim. This document (`chat_context_2.md`) extends it with job-search-specific context.
2. **HF adapter is LIVE** at `huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter` (username = `gauravgarwal`, distinct from GitHub `gauravgarwal9011`).
3. **GitHub repo name has a typo** — it's `NyayGPT`, not `NyayaGPT`. Don't auto-correct.
4. **The Blackwell cuBLAS / GGUF-only inference story is the hero narrative** for the resume + blog. Lead with it.
5. **Day 5 A/B compares vanilla Mistral Q4_K_M vs NyayaGPT Q4_K_M** — same quantization, isolating fine-tuning. Don't conflate this with the quantization benchmark (which compares variants of the fine-tuned model).
6. **Resume hasn't been compiled yet** — user will do this on Overleaf.
7. **Leaked HF token from previous session was likely rotated** (since HF adapter is now live), but explicit confirmation should still be sought.

End of context.
