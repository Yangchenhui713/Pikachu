# %% [markdown]
# # **使用 LangGraph 和 Qwen 模型实现上下文工程四大策略**
# 
# ### **目标**
# 本 Notebook 将作为一份详细的技术指南，演示如何使用 LangGraph 框架和通义千问（Qwen）模型，一步步实现“上下文工程”的四大核心策略。
# 
# 四大策略包括：
# 1.  **写入 (Write):** 为智能体构建一个“外部大脑”（暂存区），以在长任务中保持状态。
# 2.  **选择 (Select):** 使用检索增强生成（RAG）从知识库中精准调取信息。
# 3.  **压缩 (Compress):** 智能地总结对话历史，以节省成本和Token。
# 4.  **隔离 (Isolate):** 使用多智能体（Multi-agent）架构，将复杂任务分解给专家处理。
# 
# ---
# ### **第一步：环境设置与模型初始化**

# %%
# ! pip install langchain langchain-qwq langgraph pandas python-dotenv langchain_community dashscope faiss-cpu pandas

# %%
import os
import getpass
import json
import operator
from typing import List, TypedDict, Annotated

import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_qwq import ChatQwen
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# %%
# --- 1. 设置API密钥 ---
# 请注意：这里需要的是 DashScope 的 API Key
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("请输入您的DashScope API Key: ")

# %%
# --- 2. 初始化Qwen模型 ---
# 我们将使用 qwen3-32b 模型作为我们智能体的“大脑”
try:
    model = ChatQwen(model="qwen3-30b-a3b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  enable_thinking=False)
    print("Qwen 模型初始化成功！")
except Exception as e:
    print(f"模型初始化失败，请检查API Key或网络连接: {e}")

# %%
# --- 3. 初始化Token计算器 ---
# 这将帮助我们量化“压缩”策略带来的效果
encoding = tiktoken.get_encoding("cl100k_base")

# %% [markdown]
# ## **策略一：写入 (Write) - 构建智能体的“草稿纸”**
# 
# **核心思想:** 不把所有中间步骤和思考都塞进主对话历史（`messages`），而是将它们“写入”到一个独立的“暂存区”（`scratchpad`）。这可以保持主对话的清晰，并为智能体提供一个可靠的短期记忆，防止在长任务中“失忆”。
# 
# **实现:** 我们将在`AgentState`中增加一个`scratchpad`字段，并通过添加`SystemMessage`来指导模型的行为，防止无限循环。

# %%
# --- 1. 定义状态 ---
from typing_extensions import TypedDict
from typing import List, Annotated
import operator

class ToolCallRecord(TypedDict):
    step: int
    tool_name: str
    args: dict
    result: str

class WriteStrategyState(TypedDict):
    messages: Annotated[list, operator.add]
    # 结构化 scratchpad，保留完整历史
    scratchpad: dict  # {"history": List[ToolCallRecord], "final_answer": str}

# %%
# --- 2. 定义工具 ---
from langchain_core.tools import tool

@tool
def simple_calculator(operation: str, a: int, b: int) -> int:
    """一个简单的计算器工具，执行加减乘除。"""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide" and b != 0:
        return a // b
    return "无效操作"

# %%
# --- 3. 定义图的节点 ---
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END

tools = [simple_calculator]
tool_node = ToolNode(tools)
model_with_tools = model.bind_tools(tools)

def agent_with_scratchpad(state: WriteStrategyState):
    """
    Agent 节点：决定下一步动作，并更新暂存区。
    """
    print("---AGENT NODE---")
    response = model_with_tools.invoke(state['messages'])

    if response.tool_calls:
        # 暂存当前待执行的工具
        state['scratchpad']['pending_tool'] = response.tool_calls[0]
        print(f"🧠 Agent Action: Call tool `{response.tool_calls[0]['name']}` "
              f"with arguments `{response.tool_calls[0]['args']}`")
    else:
        # 全部完成，保存最终答案
        state['scratchpad']['final_answer'] = response.content
        print(f"✅ Final Answer: {response.content}")

    return {"messages": [response], "scratchpad": state['scratchpad']}

def tool_node_with_scratchpad(state: WriteStrategyState):
    """
    Tool 节点：执行工具，并把结果记录到 history。
    """
    print("---TOOL NODE---")
    last_message = state['messages'][-1]
    tool_messages = tool_node.invoke([last_message])

    # 取出待处理的工具调用
    pending = state['scratchpad']['pending_tool']
    record = ToolCallRecord(
        step=len(state['scratchpad'].get("history", [])) + 1,
        tool_name=pending['name'],
        args=pending['args'],
        result=str(tool_messages[0].content)
    )
    # 追加到历史
    state['scratchpad'].setdefault("history", []).append(record)
    print(f"📝 Recorded Tool Call: {record}")
    state['scratchpad'].pop("pending_tool", None)  # 清理

    #print(f"🛠️ Tool Result: `{record['result']}`")
    return {"messages": tool_messages, "scratchpad": state['scratchpad']}

# %%
# --- 4. 构建图 ---
from langgraph.graph import StateGraph

write_graph_builder = StateGraph(WriteStrategyState)
write_graph_builder.add_node("agent", agent_with_scratchpad)
write_graph_builder.add_node("action", tool_node_with_scratchpad)
write_graph_builder.set_entry_point("agent")

# 条件边：检查是否还有未完成的工具
def should_continue(state: WriteStrategyState) -> str:
    # 若最终答案已存在，直接结束
    if state['scratchpad'].get("final_answer"):
        return END
    return "action"

write_graph_builder.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
write_graph_builder.add_edge("action", "agent")
write_graph = write_graph_builder.compile()

# %%
# --- 5. 演示 ---
print("### 演示“写入”策略 ###")
task = (
    "1) 初始现金流 128 元与预算追加 72 元先进行合并；\n"
    "2) 合并后的资金按季度复利 3 倍杠杆放大；\n"
    "3) 放大后的资金因汇率折算需除以 100 得到基准单位值；\n"
    "4) 基准单位值再按 20 倍风险系数放大，形成风险敞口；\n"
    "5) 最终从风险敞口中一次性扣除 222 元的固定准备金。\n"
    "请列出每一步的数值结果，并以『最终结果：{数值}』的格式给出答案。"
)

system_prompt = (
    "你是一个计算助手。请按步骤使用 `simple_calculator` 工具来回答用户的问题。 "
)
initial_messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=task)
]
initial_state = {"messages": initial_messages, "scratchpad": {"history": [], "final_answer": None}}

# 使用 .stream() 观察每一步
for step in write_graph.stream(initial_state, {"recursion_limit": 20}):
    #print(step)
    print("---")

# %% [markdown]
# ## **策略二：选择 (Select) - 精准的"信息调取"**
# 
# **核心思想：** 使用RAG（检索增强生成）技术，从外部知识库中精准检索最相关的信息片段，只将必要信息注入上下文。
# 
# **实现步骤：**
# 1. 创建产品知识库（模拟向量数据库）
# 2. 构建RAG检索器
# 3. 设计智能体流程：问题 → 检索 → 生成答案
# 4. 可视化Token节省效果

# %%
# --- 1. 创建模拟产品知识库 ---
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings

# 创建嵌入模型
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

# %%
# 产品知识文档（实际应用中会从数据库加载）
product_docs = [
    Document(page_content="""机械键盘 X1 Pro 技术规格：
- 轴体：定制青轴，60g触发压力
- 连接：三模（蓝牙5.1/2.4G/USB-C）
- 电池：4000mAh，续航200小时
- 特点：热插拔轴体，PBT双色键帽，全键无冲
- 价格：699元（限时优惠599元）""", 
             metadata={"product": "机械键盘 X1 Pro", "category": "键盘"}),
    
    Document(page_content="""游戏鼠标 M800 旗舰版：
- 传感器：原相PAW3395，26000DPI
- 微动：欧姆龙光学微动，1亿次寿命
- 重量：58g（超轻量化设计）
- RGB：1680万色，10区域独立控光
- 价格：399元（套装优惠价）""", 
             metadata={"product": "游戏鼠标 M800", "category": "鼠标"}),
    
    Document(page_content="""促销邮件写作指南：
1. 标题要吸引眼球，包含优惠信息
2. 开头用痛点场景引发共鸣
3. 突出产品核心优势（性能>参数）
4. 限时优惠制造紧迫感
5. 清晰的行动召唤按钮""", 
             metadata={"doc_type": "writing_guide"}),
    
    Document(page_content="""用户偏好分析：
科技产品消费者最关注：
- 性能参数（75%用户）
- 性价比（68%用户）
- 耐用性（52%用户）
- 外观设计（48%用户）""", 
             metadata={"doc_type": "user_insight"}),
]

# 创建向量数据库
vector_db = FAISS.from_documents(product_docs, embeddings)

# %%
# --- 2. 构建RAG检索器 ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 创建检索器
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

def format_docs(docs):
    """格式化检索到的文档"""
    return "\n\n".join(f"## 来源 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

# 创建RAG提示模板
rag_template = """
你是一位专业的产品文案助手。请根据提供的背景信息回答用户问题。

<背景信息>
{context}
</背景信息>

用户问题：{question}

请用专业、简洁的语言回答，突出产品核心优势：
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

# 创建RAG链
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)


# %%
# --- 3. 设计智能体流程 ---
class SelectStrategyState(TypedDict):
    messages: List[BaseMessage]
    context: str  # 存储检索到的上下文

def retrieve_context(state: SelectStrategyState):
    """检索节点：从知识库获取相关信息"""
    print("\n--- RETRIEVE CONTEXT ---")
    last_message = state["messages"][-1].content
    
    # 执行检索
    docs = retriever.invoke(last_message)
    context = format_docs(docs)
    
    # 计算Token节省
    orig_token_count = sum(len(encoding.encode(doc.page_content)) for doc in docs)
    context_token_count = len(encoding.encode(context))
    savings = orig_token_count - context_token_count
    
    print(f"🔍 检索到 {len(docs)} 条相关文档")
    print(f"📉 Token节省: {savings} (原始: {orig_token_count} -> 压缩: {context_token_count})")
    print(f"📝 注入上下文:\n{context[:300]}...")
    
    return {"context": context}

def generate_with_context(state: SelectStrategyState):
    """生成节点：使用检索到的上下文生成回答"""
    print("\n--- GENERATE WITH CONTEXT ---")
    question = state["messages"][-1].content
    
    # 使用RAG链生成回答
    response = rag_chain.invoke(question)
    
    # 创建消息对象
    response_message = HumanMessage(content=response)
    
    # 输出结果
    print(f"💡 生成的回答: {response}")
    return {"messages": [response_message]}


# %%
# --- 4. 构建选择策略图 ---
from langgraph.graph import StateGraph

# 定义状态
class SelectState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str

# 创建图
select_graph = StateGraph(SelectState)

# 添加节点
select_graph.add_node("retrieve", retrieve_context)
select_graph.add_node("generate", generate_with_context)

# 设置入口点
select_graph.set_entry_point("retrieve")

# 添加边
select_graph.add_edge("retrieve", "generate")
select_graph.add_edge("generate", END)

# 编译图
select_workflow = select_graph.compile()

# %%
# --- 5. 演示选择策略 ---
print("\n### 演示'选择'策略 ###")
question = "请为我们的旗舰机械键盘X1 Pro写一封促销邮件，突出其核心优势"

# 初始状态
initial_state = SelectState(
    messages=[HumanMessage(content=question)],
    context=""
)

# 执行工作流
for step in select_workflow.stream(initial_state):
    if "__end__" not in step:
        print(step)
        print("---")

# %% [markdown]
# ## **策略三：压缩 (Compress) - 为上下文"瘦身减负"**
# 
# **核心思想：** 使用总结(summarization)和裁剪(trimming)技术减少上下文长度，节省Token并提高效率。
# 
# **实现两种压缩技术：**
# 1. **总结压缩**：将长文本提炼为简洁摘要
# 2. **裁剪压缩**：智能保留对话中最相关的部分
# 
# **场景演示：** 智能体需要阅读一篇长文章并回答问题，我们通过总结压缩文章内容；同时展示对话历史裁剪技术。

# %%
# --- 1. 准备长文本示例 ---
long_article = """
在人工智能领域，大语言模型（LLM）的发展正以前所未有的速度推进。2023年，OpenAI发布了GPT-4模型，其上下文窗口扩展到32K tokens，大大增强了处理长文档的能力。随后，Anthropic推出了Claude 2.1模型，支持200K tokens的上下文窗口，创下了当时的新纪录。

然而，2024年，这一纪录被中国科技公司深度求索（DeepSeek）打破。他们发布了DeepSeek-R1模型，不仅支持128K tokens的上下文窗口，还创新性地引入了"上下文压缩"技术。该技术通过智能总结和关键信息提取，可以将长文档压缩到原长度的20%-30%，同时保留95%以上的核心信息。

DeepSeek-R1的技术创新主要体现在三个方面：
1. 分层总结架构：模型首先对文档进行分段总结，然后对分段摘要进行二次总结，形成层次化的压缩结构。
2. 语义密度优化：通过强化学习训练，模型学会识别并保留信息密度最高的内容。
3. 自适应压缩率：根据用户任务类型动态调整压缩强度，平衡信息保留与效率。

在实际测试中，DeepSeek-R1处理一篇10,000字的科技论文时，将其压缩到1,500字的关键摘要，同时准确回答了论文中的核心问题。更令人印象深刻的是，压缩后的Token使用量仅为原始的18%，而任务完成质量仅下降2%。

这项技术的商业应用前景广阔：
- 法律行业：快速分析冗长的法律文件
- 金融领域：高效处理年度财报和招股书
- 学术研究：加速文献综述过程
- 客户服务：快速理解长篇客户反馈

DeepSeek团队表示，他们下一步将探索"动态上下文压缩"，即在对话过程中实时调整压缩率，进一步优化智能体的长期记忆管理。
"""

# %%
# --- 2. 定义压缩工具 ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 总结压缩工具
summary_prompt = ChatPromptTemplate.from_template(
    "请将以下文本总结为不超过{max_words}字的关键要点，保留所有核心技术和数据：\n\n{text}"
)

summarizer_chain = (
    summary_prompt
    | model
    | StrOutputParser()
)

# 裁剪压缩函数
def trim_messages(messages: List[BaseMessage], max_messages=5) -> List[BaseMessage]:
    """裁剪对话历史，保留系统消息和最新的几条消息"""
    # 始终保留第一条系统消息
    system_message = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
    
    # 保留最近的max_messages条消息（排除系统消息）
    recent_messages = messages[-max_messages:] if len(messages) > 1 else messages
    
    # 重新组合
    trimmed = []
    if system_message:
        trimmed.append(system_message)
    trimmed.extend(recent_messages)
    
    return trimmed

# %%
# --- 3. 定义状态和节点 ---
class CompressState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    original_text: str  # 原始长文本
    compressed_text: str  # 压缩后的文本
    token_savings: int  # 节省的Token数量

def compress_long_text(state: CompressState):
    """总结压缩节点：将长文本压缩为摘要"""
    print("\n--- COMPRESSING LONG TEXT ---")
    last_message = state["messages"][-1]
    
    # 从用户消息中提取问题
    question = last_message.content
    
    # 压缩长文本
    summary = summarizer_chain.invoke({"text": state["original_text"], "max_words": 300})
    
    # 计算Token节省
    orig_tokens = len(encoding.encode(state["original_text"]))
    comp_tokens = len(encoding.encode(summary))
    savings = orig_tokens - comp_tokens
    
    print(f"📉 文本压缩: {orig_tokens} tokens → {comp_tokens} tokens (节省 {savings} tokens)")
    print(f"📝 压缩摘要:\n{summary[:200]}...")
    
    # 更新状态
    return {
        "compressed_text": summary,
        "token_savings": savings,
        "messages": [HumanMessage(content=f"基于以下摘要回答问题:\n{summary}\n\n问题: {question}")]
    }

def answer_with_compressed_text(state: CompressState):
    """回答节点：基于压缩文本回答问题"""
    print("\n--- ANSWERING WITH COMPRESSED TEXT ---")
    
    # 调用模型生成答案
    response = model.invoke(state["messages"])
    answer = response.content
    
    print(f"💡 生成的回答: {answer[:200]}...")
    return {"messages": [response]}

def trim_context(state: CompressState):
    """裁剪节点：压缩对话历史"""
    print("\n--- TRIMMING CONTEXT ---")
    
    # 计算裁剪前的Token
    all_messages = "".join(m.content for m in state["messages"])
    before_tokens = len(encoding.encode(all_messages))
    
    # 执行裁剪
    trimmed_messages = trim_messages(state["messages"], max_messages=3)
    
    # 计算裁剪后的Token
    trimmed_content = "".join(m.content for m in trimmed_messages)
    after_tokens = len(encoding.encode(trimmed_content))
    savings = before_tokens - after_tokens
    
    print(f"✂️ 裁剪历史: {len(state['messages'])}条 → {len(trimmed_messages)}条消息")
    print(f"📉 Token节省: {savings} (原始: {before_tokens} -> 裁剪后: {after_tokens})")
    
    return {"messages": trimmed_messages}

# %%
# --- 4. 构建压缩策略图 ---
compress_graph = StateGraph(CompressState)

# 添加节点
compress_graph.add_node("compress", compress_long_text)
compress_graph.add_node("answer", answer_with_compressed_text)
compress_graph.add_node("trim", trim_context)

# 设置入口点
compress_graph.set_entry_point("compress")

# 添加边
compress_graph.add_edge("compress", "answer")
compress_graph.add_edge("answer", END)

# 添加条件边用于裁剪
def should_trim(state: CompressState):
    """当消息超过5条时触发裁剪"""
    if len(state["messages"]) > 5:
        return "trim"
    return END

compress_graph.add_conditional_edges("answer", should_trim, {"trim": "trim", END: END})
compress_graph.add_edge("trim", END)

# 编译图
compress_workflow = compress_graph.compile()

# %%
# --- 5. 演示总结压缩 ---
print("\n### 演示'总结压缩'技术 ###")
question = "DeepSeek-R1在文本压缩方面有哪些技术创新？压缩效果如何？"

# 初始状态
initial_state = CompressState(
    messages=[SystemMessage(content="你是一个AI技术分析师"), HumanMessage(content=question)],
    original_text=long_article,
    compressed_text="",
    token_savings=0
)

# 执行工作流
for step in compress_workflow.stream(initial_state):
    if "__end__" not in step:
        print(step)
        print("---")


# %%
# --- 6. 演示对话历史裁剪 ---
print("\n### 演示'裁剪压缩'技术 ###")

# 创建一个长对话历史
long_chat_history = [
    SystemMessage(content="你是一个专业的旅行助手"),
    HumanMessage(content="我想计划一次去日本的旅行"),
    HumanMessage(content="时间大概是明年3月下旬，10天左右"),
    HumanMessage(content="我对京都的文化景点特别感兴趣"),
    HumanMessage(content="另外也想体验一下东京的现代化都市"),
    HumanMessage(content="预算方面希望控制在2万元以内"),
    HumanMessage(content="请帮我规划一个行程"),
    HumanMessage(content="对了，我还想体验一次温泉旅馆"),
    HumanMessage(content="最好是那种传统的日式旅馆"),
    HumanMessage(content="现在请给我具体的行程建议")
]

# 初始状态（无文本压缩）
trim_demo_state = CompressState(
    messages=long_chat_history,
    original_text="",
    compressed_text="",
    token_savings=0
)

# 执行裁剪
trimmed_state = trim_context(trim_demo_state)

# 显示裁剪效果
print("\n裁剪前消息:")
for i, msg in enumerate(long_chat_history):
    prefix = "🤖" if isinstance(msg, SystemMessage) else "👤"
    print(f"{prefix} {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")

print("\n裁剪后消息:")
for i, msg in enumerate(trimmed_state['messages']):
    prefix = "🤖" if isinstance(msg, SystemMessage) else "👤"
    print(f"{prefix} {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")


# %% [markdown]
# ## **策略四：隔离 (Isolate) - "分而治之"的架构智慧**
# 
# **核心思想：** 将复杂任务分解为多个子任务，由专门的智能体在隔离环境中处理，避免上下文污染。
# 
# **实现多智能体架构：** 创建一个由主管（Supervisor）协调的专家团队（分析师+文案）。

# %%
# --- 1. 准备数据 ---
import pandas as pd
from io import StringIO

# 创建示例销售数据CSV
sales_data = """
日期,产品,销售额,销售量
2024-01-01,机械键盘,12800,32
2024-01-01,游戏鼠标,9800,49
2024-01-02,机械键盘,14500,36
2024-01-02,游戏鼠标,10200,51
2024-01-03,机械键盘,16200,40
2024-01-03,游戏鼠标,10800,54
2024-01-04,机械键盘,13800,34
2024-01-04,游戏鼠标,11200,56
2024-01-05,机械键盘,17500,42
2024-01-05,游戏鼠标,11800,59
"""

# 保存为CSV文件
with open("sales_data.csv", "w") as f:
    f.write(sales_data)

# %%
# --- 2. 定义状态 ---
# 定义多智能体协作的状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    task: str
    analysis_result: str
    final_output: str
    next_agent: str

# %%
# --- 3. 定义工具 ---
@tool
def analyze_sales_data(question: str) -> str:
    """
    分析销售数据CSV文件，找出销售额最高的产品及其销售总额。
    参数:
        question (str): 用户的原始问题，用于记录分析背景。
    返回:
        str: 一个逗号分隔的字符串，包含产品名称和总销售额，例如 "产品A,150000"。
    """
    print(f"\n--- TOOL: ANALYZE SALES DATA ---")
    print(f"📝 分析任务: {question}")
    
    try:
        df = pd.read_csv("sales_data.csv")
        product_sales = df.groupby("产品")["销售额"].sum()
        top_product = product_sales.idxmax()
        top_sales = product_sales.max()
        result = f"{top_product},{top_sales}"
        print(f"🏆 分析结果: {result}")
        return result
    except Exception as e:
        return f"分析失败: {e}"

@tool
def write_marketing_copy(product: str, key_points: str) -> str:
    """
    为指定产品撰写营销文案。
    参数:
        product (str): 需要撰写文案的产品名称。
        key_points (str): 文案需要围绕的核心卖点。
    返回:
        str: 生成的营销文案。
    """
    print(f"\n--- TOOL: WRITE MARKETING COPY ---")
    print(f"📝 撰写文案: {product} - {key_points[:50]}...")
    
    writer_prompt = ChatPromptTemplate.from_template(
        "你是一个专业营销文案。请基于以下产品信息撰写一篇不超过150字的吸引人的营销文案:\n"
        "产品名称: {product}\n"
        "核心卖点: {key_points}\n"
        "文案:"
    )
    writer_chain = writer_prompt | model | StrOutputParser()
    
    return writer_chain.invoke({"product": product, "key_points": key_points})

# %%
# --- 4. 创建智能体 ---

# Helper function to create a specialist agent
def create_agent(system_prompt: str, tools: list):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    agent = prompt | model.bind_tools(tools)
    return agent

# 分析师智能体
analyst_agent = create_agent(
    "你是一名专业的数据分析师。你的任务是分析给定的数据并返回关键结果。请使用`analyze_sales_data`工具来完成任务。",
    [analyze_sales_data]
)

# 文案智能体
writer_agent = create_agent(
    "你是一名专业的营销文案。你的任务是根据分析结果，为产品撰写引人注目的营销文案。请使用`write_marketing_copy`工具来完成任务。",
    [write_marketing_copy]
)

# --- 5. 定义智能体节点 ---

def analyst_node(state: AgentState):
    print("\n--- CALLING ANALYST AGENT ---")
    result = analyst_agent.invoke({"messages": [HumanMessage(content=state['task'])]})
    return {"messages": [result]}

def writer_node(state: AgentState):
    print("\n--- CALLING WRITER AGENT ---")
    # 从state中提取分析结果，并作为输入传递给文案智能体
    product, sales = state['analysis_result'].split(',')
    prompt = f"分析结果：销售冠军是‘{product}’，总销售额为 {sales} 元。请为此产品撰写营销文案。"
    result = writer_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"messages": [result]}

# 定义工具执行节点
tool_node = ToolNode([analyze_sales_data, write_marketing_copy])

def execute_tools(state: AgentState):
    print("\n--- EXECUTING TOOLS ---")
    last_message = state['messages'][-1]
    tool_call = last_message.tool_calls[0]
    
    # 执行工具
    tool_result = tool_node.invoke([last_message])
    
    # 根据工具更新状态
    if tool_call['name'] == 'analyze_sales_data':
        return {"messages": tool_result, "analysis_result": tool_result[0].content}
    elif tool_call['name'] == 'write_marketing_copy':
        return {"messages": tool_result, "final_output": tool_result[0].content}
    
    return {"messages": tool_result}


# --- 6. 构建图 (Supervisor模式) ---

def supervisor_router(state: AgentState):
    """路由：决定下一个应该由哪个智能体来处理"""
    print("\n--- SUPERVISOR ---")

    # 如果分析结果还未产生，则分配给分析师
    if not state.get("analysis_result"):
        print("📋 任务分配: 分析师 (Analyst)")
        return "analyst"
        
    # 如果分析已完成但文案还未撰写，则分配给文案
    if state.get("analysis_result") and not state.get("final_output"):
        print("📋 任务分配: 文案撰写 (Writer)")
        return "writer"
        
    # 如果一切都完成了
    print("✅ 所有任务完成")
    return END

# 构建图
isolate_graph = StateGraph(AgentState)

isolate_graph.add_node("analyst", analyst_node)
isolate_graph.add_node("writer", writer_node)
isolate_graph.add_node("execute_tools", execute_tools)

# 设置入口点
isolate_graph.set_entry_point("analyst")

# 定义图的边
isolate_graph.add_edge("analyst", "execute_tools")
isolate_graph.add_edge("writer", "execute_tools")
isolate_graph.add_conditional_edges(
    "execute_tools",
    supervisor_router,
    {"analyst": "analyst", "writer": "writer", END: END}
)

# 编译工作流
isolate_workflow = isolate_graph.compile()

# --- 7. 执行多智能体协作 ---
print("\n### 演示多智能体协作 (Supervisor模式) ###")
task = (
    "分析销售数据找出销售额最高的产品，"
    "然后为该产品撰写一篇吸引人的营销文案。"
)
initial_state = AgentState(
    messages=[],
    task=task,
    analysis_result="",
    final_output="",
    next_agent="analyst"
)

# 执行工作流
for step in isolate_workflow.stream(initial_state, {"recursion_limit": 10}):
    node = list(step.keys())[0]
    state = step[node]
    print(f"--- [{node}] 步骤完成 ---")
    if "final_output" in state and state["final_output"]:
        print(f"\n🎉 最终文案:\n{state['final_output']}")


