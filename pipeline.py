from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


# ============================================================
# Helper Functions
# ============================================================

def extract_text(message):
    """
    Extract clean text from a LangChain/Gemini message.

    Gemini can sometimes return content as:
    [{'type': 'text', 'text': '...'}]

    This function converts it into normal readable text.
    """

    content = getattr(message, "content", message)

    # Normal string
    if isinstance(content, str):
        return content.strip()

    # Gemini structured content
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts).strip()

    return str(content).strip()


def print_header(title):
    """Print a clean section header."""

    print("\n")
    print("=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_subheader(title):
    """Print a smaller section header."""

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# Main Research Pipeline
# ============================================================

def run_research_pipeline(topic: str):

    state = {}

    # --------------------------------------------------------
    # Project Header
    # --------------------------------------------------------

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "MULTI-AGENT AI RESEARCH SYSTEM".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print("\nResearch Topic")
    print("-" * 70)
    print(topic)

    # ========================================================
    # STEP 1 — SEARCH AGENT
    # ========================================================

    print_header("STEP 1 — SEARCH AGENT")

    print("Status: Searching for relevant information...")

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Find recent, reliable and detailed information about:

{topic}

Use the web search tool and provide:
1. A concise research summary.
2. The important findings.
3. The source titles and URLs returned by the search tool.

Do not invent URLs.
"""
            )
        ]
    })

    state["search_result"] = extract_text(
        search_result["messages"][-1]
    )

    print("Status: ✓ Search completed")

    print_subheader("SEARCH RESULTS")
    print(state["search_result"])

    # ========================================================
    # STEP 2 — READER AGENT
    # ========================================================

    print_header("STEP 2 — READER AGENT")

    print("Status: Reading the most relevant source...")

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Based on the following search results about:

{topic}

Identify the most relevant URL available in the search results.

Then use the scrape_url tool to read that URL.

Search Results:
{state["search_result"][:10000]}

Return:
1. The selected source title.
2. The selected URL.
3. The important information extracted from the page.
"""
            )
        ]
    })

    state["scraped_content"] = extract_text(
        reader_result["messages"][-1]
    )

    print("Status: ✓ Reading completed")

    print_subheader("EXTRACTED SOURCE CONTENT")
    print(state["scraped_content"])

    # ========================================================
    # STEP 3 — WRITER AGENT
    # ========================================================

    print_header("STEP 3 — WRITER AGENT")

    print("Status: Generating research report...")

    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_result']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n"
        f"{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    state["report"] = extract_text(state["report"])

    print("Status: ✓ Report generated")

    print_subheader("RESEARCH REPORT")
    print(state["report"])

    # ========================================================
    # STEP 4 — CRITIC AGENT
    # ========================================================

    print_header("STEP 4 — CRITIC AGENT")

    print("Status: Reviewing research report...")

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    state["feedback"] = extract_text(state["feedback"])

    print("Status: ✓ Review completed")

    print_subheader("CRITIC EVALUATION")
    print(state["feedback"])

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "RESEARCH COMPLETE ✓".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    return state


# ============================================================
# Run Program
# ============================================================

if __name__ == "__main__":

    topic = input("\nEnter a research topic: ").strip()

    if not topic:
        print("\nPlease enter a research topic.")
    else:
        run_research_pipeline(topic)