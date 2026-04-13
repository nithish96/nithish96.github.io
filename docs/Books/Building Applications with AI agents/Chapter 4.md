# Chapter 4: Tool Use 


While foundation models are great at chatting for hours, tools are the building blocks that empower AI agents to retrieve information, perform tasks, and interact with the environment. In the AI world, a tool is a specific capability—ranging from simple calculations to complex, multistep operations—that allows an agent to execute actual changes rather than just providing text.

Much like a doctor requires a diverse set of medical instruments to diagnose and treat patients, an AI agent requires a repertoire of tools to handle various tasks effectively.

---

## LangChain Fundamentals

Before exploring tool types, it is essential to understand the core concepts of LangChain, a popular framework for building these systems.

Core Concepts

- Chat Models: These are the engines (like GPT-4o) that process prompts. LangChain uses wrappers like `ChatOpenAI` to interact with them.
-  Messages: Interactions are structured as `HumanMessage` (user input) and `AIMessage` (model responses) to maintain context.
- The `@tool` Decorator: This is used to define external functions. It registers the function and automatically generates a schema (description of inputs and outputs) so the model knows how to use it.

Tool Invocation Workflow

1. **User Input**  - "What is the weather in NYC?"

2. **Model Decision**  - The LLM sees a `get_weather` tool is available. Instead of answering, it returns a Tool Call ID and arguments:  `{"location": "NYC"}`.

3. **Local Execution**  - Your code detects the request, runs the actual API call, and receives the result: **22°C**.

4. **Feedback Loop**  - Your code sends the **22°C** result back to the LLM.

5. **Final Answer**  - The LLM reads the result and responds to the user: "It is currently **22°C in New York**."

---

### Local Tools

Local tools are designed to run on the same machine as the agent. They are based on predefined logic and are ideal for tasks where traditional programming is more reliable than a language model.

#### Strengths and Use Cases
Local tools provide precision and predictability. They are best suited for:

- Arithmetic and complex mathematics.
- Time-zone and unit conversions.
- Calendar operations and map interactions.

#### The Importance of Metadata
Because the model relies on documentation to choose a tool, developers must:

- Use precise names: Narrowly scoped names prevent the LLM from calling the tool unnecessarily.
- Write clear descriptions: Overlapping descriptions cause model confusion.
- Define strict schemas: This helps the model understand exactly how to format the input.

#### Drawbacks
- Scalability: They can be cumbersome to deploy across multiple different agents.
- Duplication: Teams often end up reimplementing the same tools independently.
- Maintenance: Any update to the tool logic requires a full redeployment of the agent service.

---

### API-Based Tools

API-based tools enable agents to interact with external services. This allows them to access information and perform computations that are impossible to do locally.

#### Benefits of APIs
- Expanded Functionality: Agents can use weather APIs, financial data streams, or translation services without needing to be retrained.
- Real-Time Data: APIs provide the most current information, which is critical for fields like stock trading or emergency response.
- Web Browsing: Tools can be created to search the open web (e.g., via Wikipedia) to ground the agent's answers in factual, up-to-date content.

#### Design Considerations
When building API tools, focus on reliability and security. External services can fail, so agents need robust error handling and fallbacks. All communications should be secured with HTTPS, and developers must be mindful of API rate limits and data privacy laws.

---

### Plug-In Tools

Plug-in tools are modular components that can be integrated with minimal customization. They allow for rapid deployment by leveraging existing libraries and third-party services.

#### Platform Ecosystems

Most leading AI providers offer their own versions of plug-ins:

- OpenAI: Offers a vast ecosystem (Expedia, Zapier), but these are currently limited to the ChatGPT product rather than the public API.
- Anthropic (Claude): Exposes "tool use" directly through its Messages API, allowing for seamless integration of moderation or domain-specific services.
- Google (Gemini): Supports function calling via Vertex AI, letting developers declare tools in a structured configuration.
- Microsoft (Phi): Integrates tightly with Azure services like cognitive search and data processing.

#### Open Source and the Future
There is a massive growing ecosystem for open-source models. Platforms like Hugging Face provide pretrained tools, while aggregators like Glama.ai and mcp.so make "Model Context Protocol" (MCP) servers searchable.

While plug-in tools are easy to integrate, they are often general-purpose and lack the deep customization of bespoke local tools. However, as these catalogs grow, the gap between "ease of use" and "specialized capability" continues to shrink, making AI agents more versatile across industries like healthcare, finance, and education.

---

### Summary of Tool Types 

| Tool Type  | Best For                              | Setup Difficulty | Scalability |
|-------------|----------------------------------------|------------------|-------------|
| Local       | Math, String parsing                   | Low              | Low         |
| API-Based   | Real-time data (Weather, Stocks)       | Medium           | High        |
| Plug-in     | Third-party apps (Zapier)              | Easy             | High        |
| MCP         | Enterprise data / Universal use        | Medium           | Maximum     |


### The Model Context Protocol (MCP)

Custom integrations are brittle and scale poorly. To solve this, Anthropic (with support from OpenAI, Google, and Microsoft) introduced the Model Context Protocol (MCP). Think of MCP as the "USB-C port for AI"—a single, universal interface that allows any agent to connect to any data source without bespoke "glue code."



#### The Two Pillars of MCP
1.  MCP Server: A web server that exposes data (SQL databases, CRMs, etc.) via a standardized JSON-RPC 2.0 interface. It advertises a "method catalog" that tells the agent what it can do.
2.  MCP Client: The AI agent or application that "speaks" MCP. It fetches the method catalog, decides which tool to call, and sends structured requests.

#### Why It Matters
Before MCP, developers had to write unique adapters for every single system. With MCP, you implement the server once, and any MCP-capable agent can immediately discover and use those tools. Standard APIs require you to write a new "description" for every tool. MCP servers "self-describe," meaning the agent can ask the server "What can you do?" and the server hands over the manual automatically.

---

### Managing Stateful Tools

When an agent interacts with "state" (data that persists, like a database), the stakes are higher. A model might "optimize" a table by accidentally dropping all its rows.

#### Safeguards for Persistence

- Narrow Scoping: Don't give an agent "arbitrary SQL" access. Instead, register specific tools like `get_user_profile` or `add_new_customer`.
- Sanitization: Use prepared statements and reject dangerous keywords like `DROP` or `ALTER` to prevent prompt injection attacks.
- Principle of Least Privilege: If an agent only needs to read data, ensure its database credentials do not have "write" or "delete" permissions.
- Observability: Log every tool invocation. Real-time alerts for suspicious patterns allow humans to intervene before an error cascades.

---

## Automated Tool Development

We are entering a phase where foundation models no longer just use tools—they build them.

### Foundation Models as Tool Makers
By feeding an LLM an API specification (like OpenAPI), the model can draft its own wrappers and helper functions. It can execute the code in a sandbox, see the error codes, and critique itself until the tool works. This turns a sprawling, messy API landscape into a lean, agent-ready toolkit.

### Real-Time Code Generation
Some agents can write and execute code on-the-fly to solve novel problems.

- The Upside: Incredible adaptability. The agent can interface with an unfamiliar API by writing the connection logic in real time.
- The Downside: Risk and repeatability. Code generated in real time might be inefficient, insecure, or produce different results every time it runs, making debugging a nightmare.

---

## Tool Use Configuration and Reliability

Modern APIs allow you to control how "eager" an agent is to use its tools:
 
- Auto: The model decides based on context.
- Required (Any): Forces the model to use at least one tool (useful when the task -requires- an action).
- None: Blocks all tool calls for pure text generation.

### Building Production-Grade Reliability

Even with the best configuration, agents can fail. Robust systems implement a Validation -> Retry -> Fallback loop:

-  Validate: Check outputs against a schema (like Pydantic).
-  Retry: Use intelligent logic like exponential backoff for minor network blips.
-  Fallback: If all else fails, switch to a backup model, use cached data, or ask the user for help.

---

## Conclusion

The toolkit is the most critical asset of a modern AI agent. Whether you are using precise Local Tools, expansive API-based Tools, or standardized MCP Services, the goal remains the same: provide the agent with the capabilities to succeed while maintaining strict boundaries of safety and reliability.