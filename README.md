# Knowledge Engine AI Assistant 🧠

An advanced AI-powered knowledge assistant built with LangChain/LangGraph and Flask. Features a ReAct (Reasoning and Acting) agent architecture with 16+ custom tools for research, learning, productivity, and knowledge management.

## Live Demo

🔗 [Knowledge Engine AI](https://aiengine777.pythonanywhere.com/)

## Features

**Core AI Capabilities:**
- ReAct agent architecture (Reasoning + Acting)
- GPT-4o-mini powered responses
- Context-aware conversations
- Multi-tool orchestration
- Persistent memory across sessions

**Knowledge Management:**
- Wikipedia search with citations
- Save knowledge to local database
- Retrieve saved information
- Category-based organization
- Automatic timestamping

**Learning Tools:**
- 📚 Flashcard generator from any content
- 🗺️ Learning roadmap creator for any skill
- 🧠 Mind map generator
- 📖 Universal summarizer (text/books/videos)
- 📝 Step-by-step problem solver
- 💡 Idea expander framework

**Utility Tools:**
- 🧮 Safe mathematical calculator
- 💻 Simple coding assistance
- 📚 Vocabulary builder with definitions
- 🌍 Language translator (10+ languages)
- 💬 Quote finder
- 📄 Citation generator (APA/MLA/Chicago)
- ⏰ Date and time information

**Smart Features:**
- Automatic tool selection based on query
- Real-time typing indicators
- Conversation history management
- Session-based memory
- Quick action buttons

## Tech Stack

**Backend:**
- Flask 3.0+
- LangChain & LangGraph
- OpenAI GPT-4o-mini
- Python 3.8+

**AI Framework:**
- LangChain for agent orchestration
- Custom tool implementations
- ReAct agent pattern
- Message history management

**Frontend:**
- Vanilla JavaScript
- CSS3 animations
- Responsive design
- Space-themed UI

**APIs:**
- OpenAI API
- Wikipedia API
- Dictionary API
- LibreTranslate API

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/theotechtrad/AI-knowledge-engine.git
cd AI-knowledge-engine
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Setup environment variables**

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

Get your OpenAI API key: https://platform.openai.com/api-keys

**4. Run the application**
```bash
python main.py
```

Visit `http://localhost:5000`

## Project Structure

```
AI-knowledge-engine/
├── main.py                    # Flask server & AI agent (MAIN FILE)
├── frontend/
│   └── index.html            # Frontend interface (MAIN FILE)
├── .python-version           # Python version config
├── pyproject.toml            # Project config (uv)
├── uv.lock                   # Dependency lock file
├── knowledge_base.json       # Saved knowledge (auto-created)
├── flashcards.json           # Generated flashcards (auto-created)
├── mindmaps.json             # Created mind maps (auto-created)
├── ideas_journal.json        # Idea expansions (auto-created)
├── vocabulary_list.json      # Saved vocabulary (auto-created)
├── learning_roadmaps.json    # Learning paths (auto-created)
└── .env                      # API keys (create this)
```

**Note:** All `.json` files are automatically created when you use the corresponding features. You only need `main.py`, `frontend/index.html`, and `.env` to start.

## How It Works

### ReAct Agent Architecture

The system uses a ReAct (Reasoning and Acting) pattern:

1. **Reasoning:** Agent analyzes the user's query
2. **Tool Selection:** Chooses appropriate tools
3. **Action:** Executes selected tools
4. **Observation:** Processes tool results
5. **Response:** Generates final answer

### Tool System

16 specialized tools available:

**Research & Knowledge:**
- `search_wikipedia` - Search with citations
- `save_knowledge` - Store information
- `retrieve_knowledge` - Search saved data
- `list_knowledge_categories` - View all entries

**Learning:**
- `vocabulary_builder` - Define words + examples
- `create_flashcards` - Generate study cards
- `learning_roadmap` - Create skill roadmaps
- `create_mindmap` - Visual concept mapping

**Analysis:**
- `universal_summarizer` - Summarize any content
- `step_by_step_solver` - Problem-solving guide (math & coding)
- `idea_expander` - Expand seed ideas

**Utilities:**
- `calculate` - Safe math evaluation
- Simple coding help (debugging, explanations, snippets)
- `translate_text` - Multi-language translation
- `find_quotes` - Inspirational quotes
- `citation_generator` - Academic citations
- `get_current_time` - Date/time info

### Conversation Flow

```
User Query → ReAct Agent → Tool Selection → Tool Execution → Response Generation
                ↑                                                      ↓
                └────────────── Conversation History ─────────────────┘
```

## API Endpoints

**Chat:**
```
POST /api/chat
Body: { "message": "...", "session_id": "..." }
Response: { "success": true, "response": "..." }
```

**Clear History:**
```
POST /api/clear
Body: { "session_id": "..." }
Response: { "success": true }
```

**Stats:**
```
GET /api/stats
Response: { "knowledge_entries": 10, "flashcard_decks": 3, ... }
```

## Usage Examples

**Research:**
```
"Tell me about quantum entanglement"
→ Searches Wikipedia, provides citation
```

**Save Knowledge:**
```
"Save this: Python is a high-level language"
→ Stores in knowledge base with timestamp
```

**Create Flashcards:**
```
"Make flashcards from this: [paste your notes]"
→ Generates Q&A flashcards
```

**Learning Roadmap:**
```
"Create a learning path for machine learning"
→ 3-phase roadmap with timeline
```

**Summarize:**
```
"Summarize this article: [paste text]"
→ Key points + compression stats
```

**Expand Ideas:**
```
"Expand this idea: AI-powered study assistant"
→ 5W framework + action steps
```

**Coding Help:**
```
"Explain this Python code: [paste code]"
"Debug this function"
"Write a simple sorting algorithm"
→ Code explanations, debugging, simple implementations
```

## Persistent Knowledge Base

All saved data persists locally in JSON files:

- **knowledge_base.json** - User's saved information
- **flashcards.json** - Generated study cards
- **mindmaps.json** - Visual concept maps
- **ideas_journal.json** - Expanded ideas
- **vocabulary_list.json** - Learned words
- **learning_roadmaps.json** - Skill pathways

Data includes:
- Content
- Timestamps
- Categories
- Last accessed dates

## Configuration

**Agent Settings (main.py):**
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",      # Model selection
    temperature=0.7,          # Creativity level
)

agent = create_react_agent(
    llm, 
    TOOLS,
    prompt=SYSTEM_MESSAGE,
    config={"recursion_limit": 50}  # Max thinking steps
)
```

**Conversation History:**
- Keeps last 20 messages per session
- Prevents memory overflow
- Session-based isolation

## Features Explained

### Wikipedia Search with Citations
Searches Wikipedia and provides:
- Article summary
- Source title and URL
- Access date
- Proper citation format

### Universal Summarizer
Handles:
- Plain text summarization
- Book summaries (with Wikipedia lookup)
- YouTube video summaries (transcript-based)
- Adjustable length (short/medium/long)
- Key points extraction
- Compression statistics

### Learning Roadmap
Creates structured learning paths:
- 3 phases (Foundation, Building, Mastery)
- Weekly breakdown
- Goals and milestones
- Resource recommendations
- Practice exercises

### Flashcard Generator
Automatically creates study cards:
- Extracts key concepts
- Generates questions
- Provides answers
- Saves to JSON for review

## Deployment

Deployed on PythonAnywhere:

1. Upload all files
2. Create virtual environment
3. Install requirements
4. Set OPENAI_API_KEY in .env
5. Configure WSGI file
6. Set Flask app path

## Security Notes

- Safe calculation (no `eval` exploits)
- Input sanitization
- API key protection
- No database injection risks
- Session isolation

## Limitations

- Requires OpenAI API key (paid)
- Local storage only (no cloud sync)
- 50 recursion limit per query
- Session history limited to 20 messages
- Some tools require internet

## Future Improvements

- [ ] Cloud storage integration
- [ ] User authentication
- [ ] Export knowledge base (PDF/Markdown)
- [ ] Voice input support
- [ ] Mobile app version
- [ ] More languages
- [ ] Custom tool creation
- [ ] Collaborative knowledge bases

## Troubleshooting

**"Connection Error":**
- Ensure Flask server is running
- Check port 5000 is not in use
- Verify CORS settings

**"OpenAI API Error":**
- Check API key in .env
- Verify API key has credits
- Check API rate limits

**"Tool not working":**
- Check internet connection
- Verify external APIs are accessible
- Review tool-specific requirements

## Contributing

Contributions welcome! To add new tools:

1. Create tool function with `@tool` decorator
2. Add to TOOLS list
3. Update SYSTEM_MESSAGE
4. Test with agent

## License

MIT License - free to use and modify.

## Contact

Himanshu Yadav  
[GitHub](https://github.com/theotechtrad) | [LinkedIn](https://www.linkedin.com/in/hvhimanshu-yadav)

---

Built with LangChain, GPT-4o-mini & Flask | Deployed on PythonAnywhere

**Technologies:** Python, Flask, LangChain, LangGraph, OpenAI GPT-4o-mini, ReAct Architecture
