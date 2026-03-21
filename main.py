from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import requests
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
from dotenv import load_dotenv
import os
import time
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__, static_folder='frontend')
CORS(app)

print("Loading vector database...")
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
knowledge_collection = chroma_client.get_or_create_collection(name="knowledge_base", metadata={"hnsw:space": "cosine"})

print("Loading embeddings model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready!")

def get_embedding(text):
    return embedder.encode(text).tolist()

def rag_save(topic, content, category="general"):
    try:
        doc_id = str(uuid.uuid4())
        full_text = f"Topic: {topic}\nCategory: {category}\nContent: {content}"
        knowledge_collection.add(ids=[doc_id], embeddings=[get_embedding(full_text)], documents=[full_text], metadatas=[{"topic": topic, "category": category, "timestamp": datetime.now().isoformat(), "content": content}])
        try:
            with open('knowledge_base.json', 'r', encoding='utf-8') as f: kb = json.load(f)
        except: kb = {"entries": []}
        kb["entries"].append({"id": doc_id, "topic": topic, "content": content, "category": category, "timestamp": datetime.now().isoformat()})
        with open('knowledge_base.json', 'w', encoding='utf-8') as f: json.dump(kb, f, indent=2, ensure_ascii=False)
        return f"Saved '{topic}' in category '{category}' and vectorized in knowledge base!"
    except Exception as e:
        return f"Error saving: {str(e)}"

def rag_search(query, n_results=3):
    try:
        count = knowledge_collection.count()
        if count == 0: return "Knowledge base is empty. Save some knowledge first!"
        results = knowledge_collection.query(query_embeddings=[get_embedding(query)], n_results=min(n_results, count), include=["documents","metadatas","distances"])
        if not results["documents"][0]: return f"No results for '{query}'."
        out = f"Found {len(results['documents'][0])} result(s) for '{query}':\n\n"
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            relevance = round((1-dist)*100, 1)
            out += f"**{meta['topic']}** [{meta['category']}] — {relevance}% match\n{meta['content']}\n\n"
        return out
    except Exception as e:
        return f"Search error: {str(e)}"

def rag_list_categories():
    try:
        count = knowledge_collection.count()
        if count == 0: return "Knowledge base is empty!"
        all_items = knowledge_collection.get(include=["metadatas"])
        cats = {}
        for meta in all_items["metadatas"]:
            cats.setdefault(meta.get("category","general"), []).append(meta.get("topic","?"))
        result = f"📚 Knowledge Base ({count} entries)\n\n"
        for cat, topics in sorted(cats.items()):
            result += f"**{cat.upper()}** ({len(topics)} items)\n"
            for t in topics: result += f"  • {t}\n"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def rag_context(query):
    try:
        count = knowledge_collection.count()
        if count == 0: return ""
        results = knowledge_collection.query(query_embeddings=[get_embedding(query)], n_results=min(2, count), include=["metadatas","distances"])
        parts = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            if (1-dist) > 0.4:
                parts.append(f"[{meta['topic']}]: {meta['content']}")
        return "\n".join(parts)
    except: return ""

def tool_mindmap(topic):
    wiki_info = ""
    wiki_url = ""
    try:
        headers = {"User-Agent": "KnowledgeEngine/1.0"}
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":topic,"format":"json","srlimit":1},
            headers=headers, timeout=8
        )
        results = r.json().get("query",{}).get("search",[])
        if results:
            title = results[0]["title"]
            r2 = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ','_')}", headers=headers, timeout=8)
            if r2.status_code == 200:
                d = r2.json()
                wiki_info = d.get("extract","")[:300]
                wiki_url = d.get("content_urls",{}).get("desktop",{}).get("page","")
    except:
        pass

    data = None
    error_msg = ""
    try:
        wiki_ctx = f"Background about {topic}: {wiki_info}" if wiki_info else ""
        prompt = f"""You are an expert. Create a mind map for the topic: "{topic}"
{wiki_ctx}

IMPORTANT: Return ONLY a JSON object. No explanation. No markdown. Just the JSON.
The JSON must have this exact structure with 6 branches ALL specific to "{topic}":

{{"topic": "{topic}", "branches": {{"Branch1Name": {{"items": ["item1", "item2", "item3"], "detail": "3-4 sentences of detailed info specific to {topic} for this branch"}}, "Branch2Name": {{"items": ["item1", "item2", "item3"], "detail": "3-4 sentences specific to {topic}"}}, "Branch3Name": {{"items": ["item1", "item2", "item3"], "detail": "3-4 sentences specific to {topic}"}}, "Branch4Name": {{"items": ["item1", "item2", "item3"], "detail": "3-4 sentences specific to {topic}"}}, "Branch5Name": {{"items": ["item1", "item2", "item3"], "detail": "3-4 sentences specific to {topic}"}}, "Branch6Name": {{"items": ["item1", "item2", "item3"], "detail": "3-4 sentences specific to {topic}"}}}}}}

Replace Branch1Name...Branch6Name with REAL branch names for "{topic}".
Example for "Cooking": use "Ingredients", "Techniques", "Equipment", "Cuisines", "Recipes", "Nutrition"
Example for "Python": use "Basics", "Data Types", "Libraries", "Web Dev", "Data Science", "Best Practices"
Make it 100% relevant to "{topic}"."""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        print(f"[mindmap] LLM response length: {len(raw)}")

        # Clean markdown fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().strip("`").strip()

        # Find JSON object
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        data = json.loads(raw)
        print(f"[mindmap] Parsed OK, branches: {list(data.get('branches',{}).keys())}")

    except Exception as e:
        error_msg = str(e)
        print(f"[mindmap] LLM failed: {error_msg}")

    if not data:
        # Fallback with topic-aware generic structure
        data = {
            "topic": topic,
            "branches": {
                f"{topic} Basics": {"items": [f"What is {topic}", f"History of {topic}", f"Why learn {topic}"], "detail": f"The basics of {topic} cover its fundamental concepts and origins. Understanding what {topic} is and why it matters is the first step."},
                f"Core Concepts": {"items": [f"Key principle 1", f"Key principle 2", f"Key principle 3"], "detail": f"The core concepts of {topic} are the building blocks every learner must master. These form the foundation for all advanced study."},
                f"Tools & Resources": {"items": [f"Top tool for {topic}", "Online resources", "Communities"], "detail": f"The best tools and resources for learning and working with {topic}. Start with the most popular ones in the community."},
                f"Applications": {"items": [f"Real use case 1", f"Real use case 2", "Industry use"], "detail": f"{topic} is applied in many real-world scenarios. Understanding these applications motivates deeper learning."},
                f"Best Practices": {"items": ["Do this", "Avoid that", "Pro tips"], "detail": f"Following best practices in {topic} ensures quality and efficiency. Learn from experts what works and what doesn't."},
                f"Career & Growth": {"items": ["Job opportunities", "Skills to build", "Next steps"], "detail": f"Knowledge of {topic} opens many career doors. Focus on building practical skills and a strong portfolio."},
            }
        }

    links = [
        {"label": f"YouTube: {topic}", "url": f"https://www.youtube.com/results?search_query={topic.replace(' ','+')}+tutorial", "icon": "🎥"},
        {"label": f"Google: {topic}", "url": f"https://www.google.com/search?q=learn+{topic.replace(' ','+')}+guide", "icon": "🔍"},
        {"label": f"Wikipedia: {topic}", "url": wiki_url or f"https://en.wikipedia.org/wiki/{topic.replace(' ','_')}", "icon": "📖"},
        {"label": f"freeCodeCamp: {topic}", "url": f"https://www.freecodecamp.org/news/search/?query={topic.replace(' ','%20')}", "icon": "💻"},
        {"label": f"GitHub: {topic}", "url": f"https://github.com/search?q={topic.replace(' ','+')}+&sort=stars", "icon": "⚙️"},
        {"label": f"Reddit: {topic}", "url": f"https://www.reddit.com/search/?q={topic.replace(' ','+')}+&sort=top", "icon": "💬"},
    ]
    data["links"] = links
    data["wiki_summary"] = wiki_info

    try:
        with open("mindmaps.json","r") as f: maps = json.load(f)
    except: maps = {"maps":[]}
    maps["maps"].append({"topic": topic, "created": datetime.now().isoformat()})
    with open("mindmaps.json","w") as f: json.dump(maps, f, indent=2)
    return "%%MINDMAP%%" + json.dumps(data, ensure_ascii=False) + "%%ENDMINDMAP%%"

def tool_roadmap(skill):
    """Generate roadmap - returns plain formatted text, no special markers"""
    wiki_info = ""
    try:
        headers = {"User-Agent": "KnowledgeEngine/1.0"}
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":skill,"format":"json","srlimit":1},
            headers=headers, timeout=6
        )
        results = r.json().get("query",{}).get("search",[])
        if results:
            title = results[0]["title"]
            r2 = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ','_')}", headers=headers, timeout=6)
            if r2.status_code == 200:
                d = r2.json()
                if "may refer to" not in d.get("extract",""):
                    wiki_info = d.get("extract","")[:200]
    except:
        pass

    # Ask LLM for roadmap content as plain text
    try:
        wiki_ctx = f"Context: {wiki_info}\n" if wiki_info else ""
        prompt = f"""Create a learning roadmap for "{skill}".
{wiki_ctx}
Format your response EXACTLY like this (fill in content specific to {skill}):

ROADMAP_START

PHASE1_TITLE: FOUNDATIONS
PHASE1_DURATION: Weeks 1-4
PHASE1_STEPS: Step 1 | Step 2 | Step 3 | Step 4
PHASE1_DETAIL: Write 3-4 sentences about what to do in weeks 1-4 for {skill}. What to install, what to learn first, what project to build.

PHASE2_TITLE: SKILL BUILDING  
PHASE2_DURATION: Weeks 5-12
PHASE2_STEPS: Step 1 | Step 2 | Step 3 | Step 4
PHASE2_DETAIL: Write 3-4 sentences about weeks 5-12 for {skill}. Specific courses, projects to build, skills to master.

PHASE3_TITLE: MASTERY
PHASE3_DURATION: Weeks 13-24
PHASE3_STEPS: Step 1 | Step 2 | Step 3 | Step 4
PHASE3_DETAIL: Write 3-4 sentences about mastery of {skill}. Specializations, career paths, portfolio, job opportunities.

PHASE4_TITLE: FREE RESOURCES
PHASE4_DURATION: Use anytime
PHASE4_STEPS: Resource 1 | Resource 2 | Resource 3 | Resource 4
PHASE4_DETAIL: Write 3-4 sentences listing REAL specific resources for {skill}: YouTube channels, free platforms, documentation, communities.

ROADMAP_END

Replace ALL content with real specific information about {skill}."""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        print(f"[roadmap] Got response for: {skill}")

        # Parse the structured response
        phases_config = [
            {"num":"1","color":"#00d4ff","icon":"🌱"},
            {"num":"2","color":"#7c4dff","icon":"🚀"},
            {"num":"3","color":"#00e676","icon":"🏆"},
            {"num":"4","color":"#ff9100","icon":"📚"},
        ]

        phases = []
        for cfg in phases_config:
            n = cfg["num"]
            def get_field(field):
                pattern = f"PHASE{n}_{field}:"
                if pattern in raw:
                    line = raw.split(pattern)[1].split("\n")[0].strip()
                    return line
                return ""

            title = get_field("TITLE") or ["FOUNDATIONS","SKILL BUILDING","MASTERY","FREE RESOURCES"][int(n)-1]
            duration = get_field("DURATION") or ["Weeks 1-4","Weeks 5-12","Weeks 13-24","Use anytime"][int(n)-1]
            steps_raw = get_field("STEPS") or f"Learn {skill} basics|Practice daily|Build projects|Join community"
            detail = get_field("DETAIL") or f"Focus on {skill} in this phase."

            steps = [s.strip() for s in steps_raw.split("|") if s.strip()][:4]
            phases.append({
                "title": title,
                "duration": duration,
                "steps": steps,
                "detail": detail,
                "color": cfg["color"],
                "icon": cfg["icon"]
            })

    except Exception as e:
        print(f"[roadmap] Error: {e}")
        phases = [
            {"title":"FOUNDATIONS","duration":"Weeks 1-4","color":"#00d4ff","icon":"🌱",
             "steps":[f"Install {skill} tools","Learn core syntax","Practice exercises","Build first project"],
             "detail":f"Start with {skill} fundamentals. Install tools, learn syntax, practice daily, build a simple first project by week 4."},
            {"title":"SKILL BUILDING","duration":"Weeks 5-12","color":"#7c4dff","icon":"🚀",
             "steps":[f"Take a {skill} course","Build 3 projects","Join community","Read documentation"],
             "detail":f"Take a structured {skill} course. Build real projects. Engage with the community. Read official docs."},
            {"title":"MASTERY","duration":"Weeks 13-24","color":"#00e676","icon":"🏆",
             "steps":["Specialize","Build portfolio","Open source","Apply for jobs"],
             "detail":f"Specialize in one area of {skill}. Build a strong portfolio. Contribute to open source."},
            {"title":"FREE RESOURCES","duration":"Use anytime","color":"#ff9100","icon":"📚",
             "steps":["YouTube tutorials","freeCodeCamp","Official docs","Reddit community"],
             "detail":f"YouTube: search '{skill} tutorial'. Free: freeCodeCamp.org. Docs: official {skill} documentation. Community: Reddit/Discord."}
        ]

    # Build links
    links = [
        {"label": f"YouTube: {skill}", "url": f"https://www.youtube.com/results?search_query=learn+{skill.replace(' ','+')}+tutorial", "icon": "🎥"},
        {"label": f"freeCodeCamp", "url": f"https://www.freecodecamp.org/news/search/?query={skill.replace(' ','%20')}", "icon": "💻"},
        {"label": f"Coursera: {skill}", "url": f"https://www.coursera.org/search?query={skill.replace(' ','%20')}", "icon": "🎓"},
        {"label": f"GitHub: {skill}", "url": f"https://github.com/search?q={skill.replace(' ','+')}+&sort=stars", "icon": "⚙️"},
    ]

    # Save
    try:
        with open("learning_roadmaps.json","r") as f: roadmaps = json.load(f)
    except: roadmaps = {"roadmaps":[]}
    roadmaps["roadmaps"].append({"skill": skill, "created": datetime.now().isoformat()})
    with open("learning_roadmaps.json","w") as f: json.dump(roadmaps, f, indent=2)

    # Return as ROADMAP marker with JSON - but now using a simpler data structure
    import base64
    road_data = {"skill": skill, "phases": phases, "links": links, "wiki_summary": wiki_info}
    road_json = json.dumps(road_data, ensure_ascii=False)
    # Encode to base64 to avoid any JSON-in-JSON issues
    road_b64 = base64.b64encode(road_json.encode('utf-8')).decode('ascii')
    return f"ROADMAP_B64_START{road_b64}ROADMAP_B64_END"


def tool_idea_expander(idea):
    try:
        prompt = (
            f'You are a brilliant creative consultant, business strategist, and innovation expert.\n'
            + f'Someone has this idea: "{idea}"\n\n'
            + 'Analyse this idea deeply and give an INSIGHTFUL, SPECIFIC expansion. Your response format:\n\n'
            + '🌱 IDEA: [restate the idea clearly]\n'
            + '📌 ONE-LINE SUMMARY: [powerful one-sentence summary]\n'
            + '🏷️ IDEA TYPE: [business/tech/social/personal/creative/other]\n\n'
            + '---\n\n'
            + '🎯 PROBLEM THIS SOLVES:\n[Explain specifically what real problem this idea addresses and who suffers from it]\n\n'
            + '👥 TARGET AUDIENCE:\n[Describe exactly who would use/buy/benefit from this. Be specific - age, situation, needs]\n\n'
            + '---\n\n'
            + '💡 WHAT IF...? (Powerful Possibilities)\n'
            + '• [What if this was 10x bigger? Give specific scenario with numbers]\n'
            + '• [What if AI was deeply integrated? Give specific example of how]\n'
            + '• [What if applied to a different industry? Name the industry and describe exactly how]\n'
            + '• [What if made completely free? How would that change the business model and impact]\n\n'
            + '🔮 UNIQUE ANGLES OTHERS WOULD MISS:\n'
            + '• [Angle 1 - specific competitive advantage]\n'
            + '• [Angle 2 - unconventional approach]\n'
            + '• [Angle 3 - partnership or distribution angle]\n\n'
            + '🌍 SIMILAR COMPANIES/PRODUCTS TO LEARN FROM:\n[Name 2-3 real existing companies or products. Explain what this idea can learn from each one]\n\n'
            + '---\n\n'
            + '⚠️ REAL CHALLENGES TO SOLVE:\n'
            + '• Challenge 1: [specific challenge] → How to overcome: [specific solution hint]\n'
            + '• Challenge 2: [specific challenge] → How to overcome: [specific solution hint]\n'
            + '• Challenge 3: [specific challenge] → How to overcome: [specific solution hint]\n\n'
            + '💰 HOW THIS MAKES MONEY:\n[If business idea: specific pricing model, target revenue, who pays, how much. If personal/social: how to fund or sustain it]\n\n'
            + '---\n\n'
            + '🚀 EXACT NEXT STEPS:\n'
            + '1. TODAY: [one very specific action to take today]\n'
            + '2. THIS WEEK: [one specific action this week]\n'
            + '3. THIS MONTH: [one specific action this month]\n'
            + '4. 3 MONTHS: [one specific milestone to hit]\n\n'
            + '⭐ POTENTIAL SCORE: [X/10] — [2-3 sentences explaining the score, what makes it strong and what the biggest risk is]\n'
            + f'\n\nBe VERY SPECIFIC to this exact idea: "{idea}". Give real insights, real company names, real numbers. No generic templates.'
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

    except Exception as e:
        result = f"""🌱 **IDEA: {idea}**

**WHAT IF...?**
• What if this idea was 10x bigger and reached millions?
• What if AI was deeply integrated to automate the core function?
• What if this was applied to healthcare, education, or finance?

**NEXT STEPS:**
1. TODAY: Write a one-page description and share with 3 people
2. THIS WEEK: Find 5 potential users and interview them
3. THIS MONTH: Build the simplest possible prototype
4. 3 MONTHS: Validate with real users before building further"""

    try:
        with open("ideas_journal.json","r") as f: ideas = json.load(f)
    except: ideas = {"ideas":[]}
    ideas["ideas"].append({"seed": idea, "created": datetime.now().isoformat()})
    with open("ideas_journal.json","w") as f: json.dump(ideas, f, indent=2)
    return result


def tool_step_solver(problem):
    problem_lower = problem.lower()

    # Detect problem type — math/coding ONLY if clearly about math or code, not just mentioning tech words
    is_math = any(x in problem_lower for x in ["calculate", "equation", "solve for", "find x", "integral", "derivative", "% of", "square root", "factorial", "matrix"]) or (any(x in problem_lower for x in ["+", "-", "*", "/"]) and any(c.isdigit() for c in problem))
    is_code = any(x in problem_lower for x in ["write code", "write a function", "write a program", "debug this", "fix this code", "coding problem", "programming problem", "implement"]) and not any(x in problem_lower for x in ["job", "career", "learn", "getting", "interview"])

    if is_math:
        ptype = "mathematical"
        type_instruction = "This is a MATH problem. Show exact step-by-step calculations with formulas and numbers. Do NOT write code."
    elif is_code:
        ptype = "coding"
        type_instruction = "This is a CODING problem. Explain the approach step by step and include code only where needed."
    elif any(x in problem_lower for x in ["job", "career", "interview", "resume", "salary", "hired", "work", "experience", "apply", "no experience", "first job"]):
        ptype = "career"
        type_instruction = "This is a CAREER problem. Give specific, real, actionable career advice. Do NOT write code."
    elif any(x in problem_lower for x in ["friend", "relationship", "family", "love", "breakup", "conflict", "fight", "partner"]):
        ptype = "relationship"
        type_instruction = "This is a RELATIONSHIP/PERSONAL problem. Be empathetic and give balanced real-world advice. Do NOT write code."
    elif any(x in problem_lower for x in ["business", "startup", "money", "invest", "profit", "customer", "product", "market"]):
        ptype = "business"
        type_instruction = "This is a BUSINESS problem. Give strategic business thinking and practical steps. Do NOT write code."
    elif any(x in problem_lower for x in ["health", "sick", "pain", "doctor", "medicine", "diet", "exercise", "sleep"]):
        ptype = "health"
        type_instruction = "This is a HEALTH/WELLNESS problem. Give practical advice. Recommend consulting a doctor for medical issues. Do NOT write code."
    else:
        ptype = "life situation"
        type_instruction = "This is a LIFE SITUATION problem. Be empathetic, practical, and give real specific advice. Do NOT write code."

    try:
        prompt = (
            f'You are an expert advisor. Someone has this problem: "{problem}"\n'
            + f'Problem type detected: {ptype}\n'
            + f'{type_instruction}\n\n'
            + 'Analyse this problem deeply and give a DETAILED, SPECIFIC, HELPFUL step-by-step solution.\n'
            + '\nYour response format:\n'
            + '🔍 PROBLEM TYPE: [type]\n'
            + '⚡ QUICK ANSWER: [direct answer in 1-2 sentences]\n\n'
            + '📊 ANALYSIS:\n[2-3 sentences analysing WHY this is a problem and what the key challenge is]\n\n'
            + '📋 STEP-BY-STEP SOLUTION:\n'
            + 'Step 1: [Title]\n→ [Detailed action]\n💡 Why: [explanation]\n\n'
            + 'Step 2: [Title]\n→ [Detailed action]\n💡 Why: [explanation]\n\n'
            + '[Continue for as many steps as needed - minimum 4, maximum 8]\n\n'
            + '⚠️ COMMON MISTAKES TO AVOID:\n'
            + '• [Mistake 1 specific to this problem]\n'
            + '• [Mistake 2]\n\n'
            + '✅ PRO TIPS:\n'
            + '• [Expert tip 1 specific to this exact situation]\n'
            + '• [Expert tip 2]\n'
            + f'\n\nBe VERY SPECIFIC to this exact problem: "{problem}". No generic advice. Real, actionable guidance only.'
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

    except Exception as e:
        result = f"""🔍 **STEP-BY-STEP: {problem}**

**Step 1: Understand the situation clearly**
→ Define exactly what the problem is. Write it down in your own words.
💡 *Why: Clarity about the problem is 50% of the solution.*

**Step 2: Gather all relevant information**
→ What do you know? What resources do you have? What are the constraints?
💡 *Why: You cannot solve what you don\'t fully understand.*

**Step 3: Generate possible solutions**
→ Brainstorm at least 3 different approaches. Don\'t judge them yet.
💡 *Why: The first solution is rarely the best one.*

**Step 4: Choose the best approach**
→ Weigh pros and cons. Consider short and long term impact.
💡 *Why: A good decision process leads to better outcomes.*

**Step 5: Take action**
→ Execute the first concrete step today.
💡 *Why: Even imperfect action beats perfect inaction.*

✅ Remember: Most problems have solutions — focus on what you CAN control."""

    return result


def tool_summarizer(content):
    """
    Smart summarizer that handles:
    - Plain text
    - YouTube URLs (extracts transcript/info)
    - Web article URLs (fetches and reads content)
    Then summarizes + saves to RAG vector DB for follow-up questions
    """
    source_type = "text"
    source_label = "Text"
    raw_content = content.strip()

    # Detect YouTube URL
    if any(x in raw_content for x in ["youtube.com/watch", "youtu.be/"]):
        source_type = "youtube"
        source_label = "YouTube Video"
        video_id = ""
        if "watch?v=" in raw_content:
            video_id = raw_content.split("watch?v=")[1].split("&")[0].strip()
        elif "youtu.be/" in raw_content:
            video_id = raw_content.split("youtu.be/")[1].split("?")[0].strip()

        # Get video info from YouTube oEmbed API (no API key needed)
        try:
            oembed = requests.get(
                f"https://www.youtube.com/oembed?url=https://youtube.com/watch?v={video_id}&format=json",
                timeout=8
            )
            if oembed.status_code == 200:
                vdata = oembed.json()
                video_title = vdata.get("title", "Unknown Video")
                author = vdata.get("author_name", "Unknown Channel")
                raw_content = f"YouTube Video: {video_title} by {author}. Video ID: {video_id}. URL: {content.strip()}"
                source_label = f"YouTube: {video_title}"
            else:
                raw_content = f"YouTube video at: {content.strip()}"
        except:
            raw_content = f"YouTube video at: {content.strip()}"

    # Detect web article URL
    elif raw_content.startswith("http://") or raw_content.startswith("https://"):
        source_type = "url"
        source_label = "Web Article"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            r = requests.get(raw_content, headers=headers, timeout=10)
            if r.status_code == 200:
                # Extract text from HTML
                text = r.text
                # Remove script and style tags
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                # Take first 3000 chars for summarization
                raw_content = text[:3000] if len(text) > 3000 else text
                source_label = f"Article: {raw_content[:50]}..."
            else:
                raw_content = f"Could not fetch URL: {content.strip()}"
        except Exception as e:
            raw_content = f"Error fetching URL: {str(e)}"

    # Now use LLM to generate smart summary
    try:
        prompt = (
            f'You are an expert analyst. Analyze and summarize this {source_type} content:\n\n'
            + f'SOURCE: {source_label}\n'
            + f'CONTENT: {raw_content[:2000]}\n\n'
            + 'Generate a comprehensive summary with this format:\n'
            + '📋 SUMMARY:\n[3-5 sentence summary of the main content]\n\n'
            + '🔑 KEY POINTS:\n• [Key point 1]\n• [Key point 2]\n• [Key point 3]\n• [Key point 4]\n• [Key point 5]\n\n'
            + '💡 MAIN INSIGHT:\n[The single most important takeaway from this content]\n\n'
            + '❓ YOU CAN NOW ASK:\n[Suggest 3 follow-up questions the user could ask about this content]\n\n'
            + 'Be specific to the actual content. No generic filler.'
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        summary_text = response.content.strip()
    except:
        # Fallback basic summary
        sentences = [s.strip() for s in re.split(r'[.!?]+', raw_content) if len(s.strip()) > 20]
        num = min(5, len(sentences))
        step = max(1, len(sentences)//num)
        picked = [sentences[i*step] for i in range(num) if i*step < len(sentences)]
        summary_text = f"📋 SUMMARY:\n\n" + ". ".join(picked) + ".\n\n🔑 KEY POINTS:\n" + "\n".join([f"• {p[:100]}" for p in picked[:4]])

    # Save to RAG vector DB for follow-up questions
    try:
        save_result = rag_save(
            topic=f"Summary: {source_label[:50]}",
            content=raw_content[:1000],
            category="summarized_content"
        )
        rag_note = f"\n\n🧠 *Saved to knowledge base! You can now ask questions about this content.*"
    except:
        rag_note = ""

    return f"**📝 {source_label.upper()}**\n\n{summary_text}{rag_note}"


def tool_wikipedia(query):
    try:
        headers = {"User-Agent": "KnowledgeEngine/1.0 (educational project; contact@example.com)"}

        # Try up to 3 search results to skip disambiguation pages
        search_r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":query,"format":"json","srlimit":3},
            headers=headers, timeout=10
        )
        results = search_r.json().get("query",{}).get("search",[])
        if not results:
            return f"Could not find anything about '{query}' on Wikipedia."

        extract = ""
        title = ""
        page_url = ""

        for result in results:
            best_title = result["title"]
            r = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title.replace(' ','_')}",
                headers=headers, timeout=10
            )
            if r.status_code == 200:
                d = r.json()
                ext = d.get("extract", "")
                # Skip disambiguation pages (they say "may refer to")
                if "may refer to" in ext or "disambiguation" in d.get("type",""):
                    continue
                title = d.get("title", best_title)
                extract = ext
                page_url = d.get("content_urls",{}).get("desktop",{}).get("page","")
                break

        if not extract:
            # Fallback: just use first result with a link
            title = results[0]["title"]
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ','_')}"
            return f"**{title}**\n\nClick the link below to read the full Wikipedia article.\n\n[🔗 Read on Wikipedia]({page_url})"

        return f"""**{title}**

{extract}

[🔗 Read full article on Wikipedia]({page_url})"""

    except Exception as e:
        return f"Wikipedia error: {str(e)}"


def tool_analyze_image(img_data: str, question: str = "") -> str:
    """Analyze image using LLM vision - OCR, describe, summarize content."""
    try:
        import base64
        # Clean base64 data URL if needed
        if "base64," in img_data:
            header, img_data = img_data.split("base64,", 1)
            media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
        else:
            media_type = "image/jpeg"

        user_prompt = question if question and "Analyze and summarize" not in question else (
            "Please analyze this image completely:\n"
            "1. Describe what you see in detail\n"
            "2. Extract ALL text visible in the image (OCR)\n"
            "3. Summarize the main content and purpose\n"
            "4. List key information, data, or insights from the image\n"
            "5. What is the context or topic of this image?"
        )

        from langchain_core.messages import HumanMessage as HM
        msg = HM(content=[
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_data}"}}
        ])

        # Use vision-capable model
        from langchain_groq import ChatGroq
        vision_llm = ChatGroq(
            api_key=GROQ_KEYS[current_key_index % len(GROQ_KEYS)],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
            max_tokens=1000
        )
        response = vision_llm.invoke([msg])
        result = response.content.strip()

        # Save to RAG
        try:
            rag_save(
                topic=f"Image Analysis: {question[:50] if question else 'Uploaded Image'}",
                content=result[:800],
                category="image_analysis"
            )
        except: pass

        return f"🖼️ **IMAGE ANALYSIS**\n\n{result}\n\n🧠 *Saved to knowledge base for follow-up questions!*"

    except Exception as e:
        return f"Image analysis error: {str(e)}\n\nNote: Make sure your Groq API key supports vision models."



def route_message(message):
    msg = message.lower().strip()
    if any(x in msg for x in ["what time","current time","what's the time","date today","what day","what is the time"]):
        return f"🕐 {datetime.now().strftime('%A, %B %d, %Y at %I:%M:%S %p')}", "tool"
    if any(x in msg for x in ["list categor","show categor","all categories","what have i saved","my knowledge"]):
        return rag_list_categories(), "tool"
    if any(x in msg for x in ["mindmap","mind map","mind-map"]):
        topic = msg
        for kw in ["create a mindmap for:","create a mindmap for","mindmap for:","mindmap for","mindmap:"]:
            if kw in topic: topic = topic.split(kw,1)[1].strip(); break
        topic = re.sub(r"^(mindmap|mind map|mind-map)\s*","",topic,flags=re.I).strip()
        return tool_mindmap(topic.title() or "General Topic"), "tool"

    if any(x in msg for x in ["roadmap","learning path","learning plan","study plan"]):
        topic = msg
        for kw in ["create a learning roadmap for:","create a learning roadmap for","roadmap for:","roadmap for","roadmap:"]:
            if kw in topic: topic = topic.split(kw,1)[1].strip(); break
        topic = re.sub(r"^(roadmap|learning path|learning plan|study plan)\s*","",topic,flags=re.I).strip()
        return tool_roadmap(topic.title() or "General Skill"), "tool"

    if any(x in msg for x in ["create code for","create code:","write code","code for:","code creator","generate code"]):
        task = msg
        for kw in ["create code for:","create code for","create code:","code for:","write code for","generate code for"]:
            if kw in task: task = task.split(kw,1)[1].strip(); break
        return call_llm_with_rag(f"Write clean, well-commented code for: {task}", []), "llm"

    if any(x in msg for x in ["expand this idea","expand idea","idea expander"]):
        idea = msg
        for kw in ["expand this idea:","expand this idea","expand idea:"]:
            if kw in idea: idea = idea.split(kw,1)[1].strip(); break
        return tool_idea_expander(idea or message), "tool"

    if any(x in msg for x in ["step by step","step-by-step","solve step"]):
        problem = msg
        for kw in ["solve step by step:","step by step:","step by step"]:
            if kw in problem: problem = problem.split(kw,1)[1].strip(); break
        return tool_step_solver(problem or message), "tool"
    if "[IMAGE_DATA:" in message:
        # Extract base64 image and question
        if "IMAGE_DATA:" in message:
            parts = message.split("[IMAGE_DATA:")
            question = parts[0].strip()
            img_data = parts[1].rstrip("]")
            return tool_analyze_image(img_data, question), "tool"

    if any(x in msg for x in ["summarize","summarise","give me a summary","tldr"]):
        content_val = re.sub(r'^(summarize|summarise|give me a summary of|tldr)[:\s]*','',msg,flags=re.I).strip() or message
        return tool_summarizer(content_val), "tool"
    if any(x in msg for x in ["search wikipedia","wikipedia"]):
        query = re.sub(r'^(search wikipedia for|search wikipedia|wikipedia)[:\s]*','',msg,flags=re.I).strip() or message
        return tool_wikipedia(query), "tool"
    if any(x in msg for x in ["save knowledge","save this","remember this","store this"]):
        tm = re.search(r'topic[=:\s]+([^,\n]+)',message,re.I)
        cm = re.search(r'content[=:\s]+([^,\n]+)',message,re.I)
        catm = re.search(r'category[=:\s]+([^,\n]+)',message,re.I)
        return rag_save(tm.group(1).strip() if tm else "General", cm.group(1).strip() if cm else message, catm.group(1).strip() if catm else "general"), "tool"
    if any(x in msg for x in ["retrieve","find knowledge","what do i know about","recall"]):
        query = re.sub(r'^(retrieve knowledge about|retrieve|find knowledge about|what do i know about|recall)[:\s]*','',msg,flags=re.I).strip() or message
        return rag_search(query), "tool"
    return None, "llm"

GROQ_KEYS = [k for k in [os.getenv("GROQ_API_KEY_1"),os.getenv("GROQ_API_KEY_2"),os.getenv("GROQ_API_KEY_3"),os.getenv("GROQ_API_KEY_4"),os.getenv("GROQ_API_KEY")] if k]
current_key_index = 0

def get_llm():
    return ChatGroq(api_key=GROQ_KEYS[current_key_index % len(GROQ_KEYS)], model="llama-3.3-70b-versatile", temperature=0.4, max_tokens=800)

llm = get_llm()
conversation_sessions = {}

def call_llm_with_rag(user_message, history):
    global current_key_index, llm
    rag_ctx = rag_context(user_message)
    system_content = "You are Knowledge Engine AI — smart, helpful assistant. Write code in ```language blocks```. Be concise."
    if rag_ctx:
        system_content += f"\n\n📚 RELEVANT FROM USER'S KNOWLEDGE BASE:\n{rag_ctx}\nUse this to enhance your answer."
    messages = [SystemMessage(content=system_content)] + history + [HumanMessage(content=user_message)]
    for attempt in range(max(len(GROQ_KEYS)*2, 2)):
        try:
            return llm.invoke(messages).content
        except Exception as e:
            if any(x in str(e).lower() for x in ["rate","429","limit","exceeded","decommission"]):
                current_key_index += 1
                llm = get_llm()
                time.sleep(2)
                continue
            raise e
    return "All API keys rate limited. Please wait a few minutes."

@app.route('/')
def index(): return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename): return send_from_directory('frontend', filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message','').strip()
        session_id = data.get('session_id','default')
        if not user_message: return jsonify({'error':'No message'}), 400
        if session_id not in conversation_sessions: conversation_sessions[session_id] = []
        history = conversation_sessions[session_id]
        tool_result, route_type = route_message(user_message)
        response_text = tool_result if route_type == "tool" else call_llm_with_rag(user_message, history)
        conversation_sessions[session_id].extend([HumanMessage(content=user_message), AIMessage(content=response_text)])
        if len(conversation_sessions[session_id]) > 6:
            conversation_sessions[session_id] = conversation_sessions[session_id][-6:]
        # Check if response contains visual markers - send raw to avoid double-encoding
        response_type = 'text'
        if '%%ROADMAP%%' in response_text or 'ROADMAP_B64_START' in response_text:
            response_type = 'roadmap'
        elif '%%MINDMAP%%' in response_text:
            response_type = 'mindmap'

        return jsonify({'response': response_text, 'success': True, 'route': route_type, 'response_type': response_type})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    data = request.json
    sid = data.get('session_id','default')
    if sid in conversation_sessions: conversation_sessions[sid] = []
    return jsonify({'success': True})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = {'knowledge_entries':0,'mindmaps':0,'ideas':0,'roadmaps':0,'vector_db_count':0}
    try: stats['vector_db_count'] = knowledge_collection.count()
    except: pass
    for key,file,field in [('knowledge_entries','knowledge_base.json','entries'),('mindmaps','mindmaps.json','maps'),('ideas','ideas_journal.json','ideas'),('roadmaps','learning_roadmaps.json','roadmaps')]:
        try:
            with open(file,'r') as f: stats[key] = len(json.load(f).get(field,[]))
        except: pass
    return jsonify({'success':True,'stats':stats})

if __name__ == '__main__':
    print("="*60)
    # Avoid Unicode/emoji in some Windows consoles (cp1252)
    print("KNOWLEDGE ENGINE - PRODUCTION MODE")
    print(f"{len(GROQ_KEYS)} API key(s) | RAG enabled | Vector DB ready")
    print("Server: http://localhost:5500")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5500)