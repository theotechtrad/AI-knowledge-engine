from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from typing import List
import json
import requests
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import re
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__, static_folder='frontend')
CORS(app)

# ========== ALL KNOWLEDGE ENGINE TOOLS ==========

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information on a topic with citation and source link."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            extract = data.get('extract', 'No information found.')
            title = data.get('title', query)
            page_url = data.get('content_urls', {}).get('desktop', {}).get('page', 'Wikipedia')
            
            citation = f"""**{title}**

{extract}

📚 **Citation:**
Source: Wikipedia
Title: {title}
URL: {page_url}
Accessed: {datetime.now().strftime('%Y-%m-%d')}

[Read Full Article]({page_url})
"""
            return citation
        else:
            return f"Could not find information about '{query}' on Wikipedia."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


@tool
def save_knowledge(topic: str, content: str, category: str = "general") -> str:
    """Save information to the knowledge base with timestamp."""
    try:
        try:
            with open('knowledge_base.json', 'r', encoding='utf-8') as f:
                kb = json.load(f)
        except FileNotFoundError:
            kb = {"entries": []}
        
        entry = {
            "id": len(kb["entries"]) + 1,
            "topic": topic,
            "content": content,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        
        kb["entries"].append(entry)
        
        with open('knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)
        
        return f"✓ Successfully saved knowledge about '{topic}' in category '{category}'. Entry ID: {entry['id']}"
    except Exception as e:
        return f"Error saving knowledge: {str(e)}"


@tool
def retrieve_knowledge(query: str) -> str:
    """Search and retrieve information from the knowledge base."""
    try:
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        query_lower = query.lower()
        matches = []
        
        for entry in kb["entries"]:
            if (query_lower in entry["topic"].lower() or 
                query_lower in entry["content"].lower() or
                query_lower in entry["category"].lower()):
                matches.append(entry)
                entry["last_accessed"] = datetime.now().isoformat()
        
        if matches:
            with open('knowledge_base.json', 'w', encoding='utf-8') as f:
                json.dump(kb, f, indent=2, ensure_ascii=False)
            
            result = f"Found {len(matches)} entries:\n\n"
            for entry in matches:
                result += f"**{entry['topic']}** (Category: {entry['category']})\n"
                result += f"{entry['content']}\n"
                result += f"Saved: {entry['timestamp'][:10]}\n\n"
            return result
        else:
            return f"No knowledge found for '{query}' in the knowledge base."
    except FileNotFoundError:
        return "Knowledge base is empty. Start by saving some information!"
    except Exception as e:
        return f"Error retrieving knowledge: {str(e)}"


@tool
def list_knowledge_categories() -> str:
    """List all categories and topics in the knowledge base."""
    try:
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        if not kb["entries"]:
            return "Knowledge base is empty."
        
        categories = {}
        for entry in kb["entries"]:
            cat = entry["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry["topic"])
        
        result = f"📚 Knowledge Base ({len(kb['entries'])} total entries)\n\n"
        for cat, topics in sorted(categories.items()):
            result += f"**{cat.upper()}** ({len(topics)} entries)\n"
            for topic in topics:
                result += f"  • {topic}\n"
            result += "\n"
        
        return result
    except FileNotFoundError:
        return "Knowledge base is empty."
    except Exception as e:
        return f"Error listing knowledge: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations safely."""
    try:
        import math
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'log': math.log, 'log10': math.log10,
            'pi': math.pi, 'e': math.e
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}\n\nCalculation: {expression} = {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}\nPlease check your expression."


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current Date & Time: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')}"


@tool
def vocabulary_builder(word: str, action: str = "define") -> str:
    """Build vocabulary: define words, find synonyms, get usage examples."""
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()[0]
            result = f"📖 **{word.upper()}**\n\n"
            
            if 'phonetic' in data:
                result += f"🔊 Pronunciation: {data['phonetic']}\n\n"
            
            for meaning in data.get('meanings', []):
                pos = meaning.get('partOfSpeech', 'unknown')
                result += f"**{pos.capitalize()}:**\n"
                
                for idx, definition in enumerate(meaning.get('definitions', [])[:3], 1):
                    result += f"{idx}. {definition.get('definition', '')}\n"
                    if 'example' in definition:
                        result += f"   💡 Example: {definition['example']}\n"
                    if 'synonyms' in definition and definition['synonyms']:
                        syns = ', '.join(definition['synonyms'][:5])
                        result += f"   🔄 Synonyms: {syns}\n"
                result += "\n"
            
            if action == "save":
                try:
                    with open('vocabulary_list.json', 'r') as f:
                        vocab = json.load(f)
                except FileNotFoundError:
                    vocab = {"words": []}
                
                vocab["words"].append({
                    "word": word,
                    "definition": data['meanings'][0]['definitions'][0]['definition'],
                    "added": datetime.now().isoformat()
                })
                
                with open('vocabulary_list.json', 'w') as f:
                    json.dump(vocab, f, indent=2)
                
                result += "\n✅ Added to your vocabulary list!"
            
            return result
        else:
            return f"Could not find definition for '{word}'."
    except Exception as e:
        return f"Error looking up word: {str(e)}"


@tool
def universal_summarizer(content: str, content_type: str = "text", summary_length: str = "medium") -> str:
    """Universal summarizer for text, books, videos, articles, and documents."""
    
    if content_type == "youtube" or "youtube.com" in content or "youtu.be" in content:
        return f"""🎥 **VIDEO SUMMARY:**

I would summarize this YouTube video, but this requires the YouTube Transcript API.
For now, here's what I can help with:

**Manual Summary Steps:**
1. Watch the video at 1.5x-2x speed
2. Note key timestamps
3. Extract main points
4. Paste the transcript here for auto-summary

**Video URL:** {content}

💡 **Tip:** Use YouTube's auto-generated captions → copy transcript → paste here for summarization!
"""
    
    elif content_type == "book":
        return f"""📚 **BOOK SUMMARY: {content}**

To provide a comprehensive book summary, I'll search for key information:

**Structure:**
├─ 📖 Main Theme/Central Idea
├─ 👥 Key Characters/Figures
├─ 💡 Core Concepts (3-5 main ideas)
├─ 🎯 Key Takeaways
└─ ⭐ Notable Quotes

💾 **Would you like me to search Wikipedia for "{content}" to get more details?**
"""
    
    else:
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "Content too short to summarize."
        
        if summary_length == "short":
            num_sentences = min(3, len(sentences))
        elif summary_length == "medium":
            num_sentences = min(6, len(sentences))
        else:
            num_sentences = min(10, len(sentences))
        
        if len(sentences) <= num_sentences:
            summary_sentences = sentences
        else:
            indices = set([0])
            indices.add(len(sentences) - 1)
            
            step = max(1, len(sentences) // (num_sentences - 2))
            for i in range(1, num_sentences - 1):
                idx = min(i * step, len(sentences) - 2)
                indices.add(idx)
            
            summary_sentences = [sentences[i] for i in sorted(indices)]
        
        summary = ". ".join(summary_sentences) + "."
        
        key_points = []
        for sent in summary_sentences[:5]:
            words = sent.split()
            if len(words) > 5:
                key_points.append(sent[:100] + "..." if len(sent) > 100 else sent)
        
        result = f"""📝 **SUMMARY ({summary_length.upper()}):**

{summary}

---
**📌 KEY POINTS:**
"""
        for idx, point in enumerate(key_points, 1):
            result += f"{idx}. {point}\n"
        
        result += f"""
---
**📊 STATISTICS:**
• Original: {len(content)} characters, {len(sentences)} sentences
• Summary: {len(summary)} characters, {len(summary_sentences)} sentences
• Compression: {len(summary)/len(content)*100:.1f}%
"""
        return result


@tool
def create_mindmap(topic: str, notes: str) -> str:
    """Convert notes into a visual mind map structure (text-based tree)."""
    lines = [line.strip() for line in notes.split('\n') if line.strip()]
    
    concepts = []
    for line in lines:
        if len(line) > 10:
            concepts.append(line[:80])
    
    mindmap = f"""🧠 **MIND MAP: {topic.upper()}**

                    ┌─────────────────┐
                    │   {topic[:15]:^15}   │
                    └─────────────────┘
                            │
            ┌───────────────┼───────────────┐
"""
    
    for i, concept in enumerate(concepts[:6], 1):
        branch = f"    [{i}] {concept}"
        mindmap += f"{branch}\n"
        
        words = concept.split()
        if len(words) > 5:
            sub_concepts = [' '.join(words[j:j+3]) for j in range(0, len(words), 3)][:2]
            for sub in sub_concepts:
                mindmap += f"         └─► {sub}\n"
    
    mindmap += f"""
---
💡 **Pro Tip:** Use this structure to visualize relationships!
📊 Total branches: {len(concepts)}
"""
    
    try:
        with open('mindmaps.json', 'r') as f:
            maps = json.load(f)
    except FileNotFoundError:
        maps = {"maps": []}
    
    maps["maps"].append({
        "topic": topic,
        "created": datetime.now().isoformat(),
        "structure": mindmap
    })
    
    with open('mindmaps.json', 'w') as f:
        json.dump(maps, f, indent=2)
    
    return mindmap


@tool
def translate_text(text: str, target_language: str) -> str:
    """Translate text to another language using LibreTranslate API."""
    try:
        url = "https://libretranslate.de/translate"
        
        payload = {
            "q": text,
            "source": "en",
            "target": target_language,
            "format": "text"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            translated = data.get('translatedText', 'Translation failed')
            
            lang_names = {
                'es': 'Spanish', 'fr': 'French', 'de': 'German', 
                'hi': 'Hindi', 'zh': 'Chinese', 'ja': 'Japanese',
                'ar': 'Arabic', 'pt': 'Portuguese', 'ru': 'Russian',
                'it': 'Italian', 'ko': 'Korean'
            }
            
            lang_name = lang_names.get(target_language, target_language.upper())
            
            return f"""🌍 **Translation to {lang_name}:**

{translated}

---
**Original (English):** {text}
**Target Language:** {lang_name} ({target_language})
"""
        else:
            return "Translation failed. Try: es (Spanish), fr (French), de (German), hi (Hindi), zh (Chinese)"
    except Exception as e:
        return f"Error translating: {str(e)}"


@tool
def idea_expander(seed_idea: str) -> str:
    """Take a seed idea and expand it with questions, perspectives, and action steps."""
    expansion = f"""🌱 **SEED IDEA:** {seed_idea}

🚀 **IDEA EXPANSION FRAMEWORK:**

**1. WHAT IF...? (Possibilities)**
   • What if this idea was 10x bigger?
   • What if it was applied to a different field?
   • What if technology/AI enhanced it?

**2. WHO, WHAT, WHERE, WHEN, WHY? (5W Framework)**
   • WHO would benefit most from this?
   • WHAT problem does it solve?
   • WHERE can this be implemented?
   • WHEN is the best time to start?
   • WHY is this important now?

**3. POTENTIAL APPLICATIONS:**
   • Business/Commercial use
   • Educational application
   • Social impact potential
   • Personal development angle

**4. CHALLENGES TO CONSIDER:**
   • What obstacles might you face?
   • What resources are needed?
   • What skills must be developed?

**5. NEXT STEPS:**
   • Research similar ideas
   • Create a simple prototype/MVP
   • Talk to 5 people about it
   • Write a one-page plan

💡 **Action Item:** Pick ONE next step and do it today!
"""
    
    try:
        with open('ideas_journal.json', 'r') as f:
            ideas = json.load(f)
    except FileNotFoundError:
        ideas = {"ideas": []}
    
    ideas["ideas"].append({
        "seed": seed_idea,
        "expanded": expansion,
        "created": datetime.now().isoformat(),
        "status": "exploring"
    })
    
    with open('ideas_journal.json', 'w') as f:
        json.dump(ideas, f, indent=2)
    
    return expansion


@tool
def create_flashcards(topic: str, content: str, num_cards: int = 5) -> str:
    """Generate study flashcards from notes or content."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', content) if len(s.strip()) > 20]
    
    if not sentences:
        return "Content too short to create flashcards."
    
    flashcards = []
    
    for i, sentence in enumerate(sentences[:num_cards], 1):
        words = sentence.split()
        
        if len(words) > 5:
            if any(word in sentence.lower() for word in ['is', 'are', 'was', 'were']):
                question = f"What {sentence.split('is', 1)[0].strip()} is?"
            elif 'because' in sentence.lower():
                parts = sentence.split('because', 1)
                question = f"Why {parts[0].strip()}?"
            else:
                question = f"Explain: {' '.join(words[:6])}..."
            
            answer = sentence
            
            flashcards.append({
                "id": i,
                "question": question,
                "answer": answer
            })
    
    result = f"""🎴 **FLASHCARD SET: {topic.upper()}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    
    for card in flashcards:
        result += f"""╔═══ CARD #{card['id']} ═══╗

❓ **QUESTION:**
{card['question']}

✅ **ANSWER:**
{card['answer']}

═══════════════════════════════════

"""
    
    result += f"""📚 **STUDY TIPS:**
• Cover the answer and try to recall
• Review cards daily for best retention
• Shuffle the order when practicing

💾 **Total Cards Generated:** {len(flashcards)}
"""
    
    try:
        with open('flashcards.json', 'r') as f:
            all_cards = json.load(f)
    except FileNotFoundError:
        all_cards = {"decks": []}
    
    all_cards["decks"].append({
        "topic": topic,
        "cards": flashcards,
        "created": datetime.now().isoformat(),
        "total_cards": len(flashcards)
    })
    
    with open('flashcards.json', 'w') as f:
        json.dump(all_cards, f, indent=2)
    
    return result


@tool
def learning_roadmap(skill: str, current_level: str = "beginner") -> str:
    """Create a comprehensive learning path/roadmap for any skill."""
    
    roadmap = f"""🎯 **LEARNING ROADMAP: {skill.upper()}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 **Current Level:** {current_level.capitalize()}
⏰ **Estimated Timeline:** 3-6 months (with consistent practice)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📚 PHASE 1: FOUNDATIONS (Weeks 1-4)

**Goals:**
├─ Understand basic concepts and terminology
├─ Build foundational knowledge
└─ Complete first small project

**Learning Path:**
1️⃣ **Week 1-2:** Core Concepts
   • Learn fundamental principles
   • Watch intro tutorials (YouTube/Coursera)
   • Practice 30 min daily

2️⃣ **Week 3-4:** Hands-On Practice
   • Complete 3-5 small exercises
   • Join online communities
   • Build first mini-project

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🚀 PHASE 2: SKILL BUILDING (Weeks 5-12)

**Goals:**
├─ Master intermediate techniques
├─ Build portfolio projects
└─ Learn best practices

**Learning Path:**
3️⃣ **Week 5-8:** Intermediate Skills
   • Deep dive into advanced topics
   • Take online course
   • Practice daily (1 hour minimum)

4️⃣ **Week 9-12:** Project Development
   • Build 2-3 portfolio projects
   • Get feedback from community
   • Refine your skills

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🎓 PHASE 3: MASTERY (Weeks 13-24)

**Goals:**
├─ Achieve professional proficiency
├─ Specialize in specific area
└─ Teach/mentor others

💡 **PRO TIPS:**
✅ Consistency over intensity - Daily practice beats weekend cramming
✅ Build projects - Learning by doing is most effective
✅ Join communities - Learn from others, ask questions

🎉 **Remember:** Everyone starts as a beginner!

💾 **Next Step:** Choose ONE resource from Phase 1 and start TODAY!
"""
    
    try:
        with open('learning_roadmaps.json', 'r') as f:
            roadmaps = json.load(f)
    except FileNotFoundError:
        roadmaps = {"roadmaps": []}
    
    roadmaps["roadmaps"].append({
        "skill": skill,
        "level": current_level,
        "created": datetime.now().isoformat(),
        "roadmap": roadmap
    })
    
    with open('learning_roadmaps.json', 'w') as f:
        json.dump(roadmaps, f, indent=2)
    
    return roadmap


@tool
def find_quotes(topic: str, num_quotes: int = 5) -> str:
    """Find inspirational and relevant quotes on any topic."""
    
    quote_database = {
        "learning": [
            ("The more that you read, the more things you will know.", "Dr. Seuss"),
            ("Education is the most powerful weapon which you can use to change the world.", "Nelson Mandela"),
            ("Live as if you were to die tomorrow. Learn as if you were to live forever.", "Mahatma Gandhi"),
        ],
        "motivation": [
            ("The only way to do great work is to love what you do.", "Steve Jobs"),
            ("Believe you can and you're halfway there.", "Theodore Roosevelt"),
            ("Success is not final, failure is not fatal: it is the courage to continue that counts.", "Winston Churchill"),
        ],
        "success": [
            ("Success is not the key to happiness. Happiness is the key to success.", "Albert Schweitzer"),
            ("The way to get started is to quit talking and begin doing.", "Walt Disney"),
        ],
    }
    
    topic_lower = topic.lower()
    matching_quotes = []
    
    for category, quotes in quote_database.items():
        if topic_lower in category or category in topic_lower:
            matching_quotes = quotes
            break
    
    if not matching_quotes:
        matching_quotes = quote_database["learning"]
    
    selected_quotes = matching_quotes[:num_quotes]
    
    result = f"""💬 **INSPIRATIONAL QUOTES: {topic.upper()}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    
    for idx, (quote, author) in enumerate(selected_quotes, 1):
        result += f"""{idx}. "{quote}"
   — {author}

"""
    
    result += """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 **Available Categories:**
• Learning • Motivation • Success • Knowledge • Creativity • Wisdom
"""
    
    return result


@tool
def step_by_step_solver(problem: str, problem_type: str = "general") -> str:
    """Solve problems step-by-step with detailed explanations."""
    
    solution = f"""🔍 **STEP-BY-STEP SOLVER**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 **PROBLEM:** {problem}
🎯 **TYPE:** {problem_type.upper()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    
    if problem_type == "math":
        solution += """**STEP 1: IDENTIFY THE PROBLEM TYPE**
├─ Read the problem carefully
├─ Identify what is given
└─ Determine what needs to be found

**STEP 2: WRITE DOWN KNOWN VALUES**
├─ List all given information
└─ Note any formulas that might apply

**STEP 3: SOLVE STEP-BY-STEP**
├─ Show each calculation clearly
└─ Simplify as you go

**STEP 4: VERIFY YOUR ANSWER**
├─ Check if the answer makes sense
└─ Review all steps for errors
"""
    
    elif problem_type == "coding":
        solution += """**STEP 1: UNDERSTAND THE REQUIREMENT**
├─ What is the input?
├─ What is the expected output?
└─ What edge cases exist?

**STEP 2: PLAN THE ALGORITHM**
├─ Break problem into smaller functions
└─ Identify data structures needed

**STEP 3: IMPLEMENT THE CODE**
├─ Start with a simple version
└─ Add comments for clarity

**STEP 4: TEST YOUR CODE**
├─ Test with normal cases
└─ Test edge cases
"""
    
    else:
        solution += """**STEP 1: UNDERSTAND THE GOAL**
├─ What are you trying to achieve?
└─ What resources do you have?

**STEP 2: CREATE A PLAN**
├─ Break into manageable steps
└─ Prioritize tasks

**STEP 3: EXECUTE SYSTEMATICALLY**
├─ Follow your plan in order
└─ Stay flexible for adjustments

**STEP 4: REVIEW & IMPROVE**
├─ Evaluate the final result
└─ Note areas for improvement
"""
    
    solution += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 **PRO TIPS:**
• ✅ Take your time on each step
• ✅ Show all your work
• ✅ Double-check calculations/logic

📚 **RESOURCES:**
• For math: Khan Academy, Wolfram Alpha
• For coding: LeetCode, Stack Overflow
"""
    
    return solution


@tool
def citation_generator(source_type: str = "website", author: str = "Unknown Author", 
                      title: str = "Untitled", year: str = "", url: str = "") -> str:
    """Generate proper citations in APA, MLA, or Chicago format."""
    if not year:
        year = str(datetime.now().year)
    
    citations = f"""📚 **CITATION GENERATOR**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Source Type:** {source_type.capitalize()}

**APA STYLE:**
{author}. ({year}). {title}. Retrieved from {url}

**MLA STYLE:**
{author}. "{title}." {year}. Web. {datetime.now().strftime('%d %b. %Y')}.

**CHICAGO STYLE:**
{author}. "{title}." Accessed {datetime.now().strftime('%B %d, %Y')}. {url}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Copy the style required by your institution!
"""
    
    return citations


# ========== ALL TOOLS LIST ==========
TOOLS = [
    search_wikipedia,
    save_knowledge,
    retrieve_knowledge,
    list_knowledge_categories,
    calculate,
    get_current_time,
    vocabulary_builder,
    universal_summarizer,
    create_mindmap,
    translate_text,
    idea_expander,
    create_flashcards,
    learning_roadmap,
    find_quotes,
    step_by_step_solver,
    citation_generator
]


# ========== MODEL SETUP ==========
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.7
)


SYSTEM_MESSAGE = """You are Knowledge Engine, an advanced AI knowledge assistant with comprehensive learning capabilities:

🔍 **CORE FEATURES:**
- Wikipedia search with citations
- Knowledge base management
- Mathematical calculations
- Vocabulary builder

📚 **ADVANCED FEATURES:**
- Universal Summarizer (text/books/videos)
- Flashcard Generator
- Learning Roadmap Creator
- Step-by-Step Problem Solver
- Mind Map Generator
- Language Translation (10+ languages)
- Idea Expander
- Quote Finder
- Citation Generator

Be conversational, helpful, and always cite your sources!"""

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)

# Store conversation history per session
conversation_sessions = {}


# ========== FLASK ROUTES ==========

@app.route('/')
def index():
    """Serve the frontend HTML file"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files like images from the frontend folder"""
    return send_from_directory('frontend', filename)



@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from frontend"""
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get or create conversation history for this session
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        history = conversation_sessions[session_id]
        
        # Run the agent
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_message)]},
            config={"recursion_limit": 50}
        )
        
        ai_response = result["messages"][-1]
        
        # Update conversation history
        conversation_sessions[session_id].extend([
            HumanMessage(content=user_message),
            ai_response
        ])
        
        # Keep only last 20 messages to avoid memory issues
        if len(conversation_sessions[session_id]) > 20:
            conversation_sessions[session_id] = conversation_sessions[session_id][-20:]
        
        return jsonify({
            'response': ai_response.content,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in conversation_sessions:
        conversation_sessions[session_id] = []
    
    return jsonify({'success': True, 'message': 'History cleared'})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about knowledge base"""
    try:
        stats = {
            'knowledge_entries': 0,
            'flashcard_decks': 0,
            'mindmaps': 0,
            'ideas': 0,
            'vocabulary_words': 0
        }
        
        # Count knowledge base entries
        try:
            with open('knowledge_base.json', 'r') as f:
                kb = json.load(f)
                stats['knowledge_entries'] = len(kb.get('entries', []))
        except FileNotFoundError:
            pass
        
        # Count flashcard decks
        try:
            with open('flashcards.json', 'r') as f:
                fc = json.load(f)
                stats['flashcard_decks'] = len(fc.get('decks', []))
        except FileNotFoundError:
            pass
        
        # Count mindmaps
        try:
            with open('mindmaps.json', 'r') as f:
                mm = json.load(f)
                stats['mindmaps'] = len(mm.get('maps', []))
        except FileNotFoundError:
            pass
        
        # Count ideas
        try:
            with open('ideas_journal.json', 'r') as f:
                ideas = json.load(f)
                stats['ideas'] = len(ideas.get('ideas', []))
        except FileNotFoundError:
            pass
        
        # Count vocabulary words
        try:
            with open('vocabulary_list.json', 'r') as f:
                vocab = json.load(f)
                stats['vocabulary_words'] = len(vocab.get('words', []))
        except FileNotFoundError:
            pass
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


if __name__ == '__main__':
    # Create frontend directory if it doesn't exist
    if not os.path.exists('frontend'):
        os.makedirs('frontend')
        print("📁 Created 'frontend' directory")
        print("⚠️  Please place your index.html file in the 'frontend' folder")
    
    print("=" * 80)
    print("🧠 KNOWLEDGE ENGINE - Flask Server Starting...")
    print("=" * 80)
    print("✅ Server running at: http://localhost:5000")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print()
    print("📚 Available Features:")
    print("  • Wikipedia Search with Citations")
    print("  • Vocabulary Builder & Dictionary")
    print("  • Text/Book/Video Summarizer")
    print("  • Flashcard Generator")
    print("  • Learning Roadmap Creator")
    print("  • Step-by-Step Problem Solver")
    print("  • Mind Map Generator")
    print("  • Language Translator (10+ languages)")
    print("  • Idea Expander")
    print("  • Quote Finder")
    print("  • Citation Generator")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)