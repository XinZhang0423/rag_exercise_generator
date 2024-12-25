from functools import wraps
from typing import Callable, Dict

class PromptsRegistry:
    """
    一个用于注册和管理所有提示的类。
    """
    _registry: Dict[str, Callable[[], str]] = {}

    @classmethod
    def register(cls, name: str):
        """
        装饰器，用于注册一个新的提示。
        """
        def decorator(func: Callable[[], str]):
            cls._registry[name] = func
            @wraps(func)
            def wrapper():
                return func()
            return wrapper
        return decorator

    @classmethod
    def get_prompt(cls, name: str) -> str:
        """
        获取指定名称的提示模板。
        """
        prompt_func = cls._registry.get(name)
        if not prompt_func:
            raise ValueError(f"Prompt '{name}' not found in registry.")
        return prompt_func()

    @classmethod
    def assemble(cls, name: str, **kwargs) -> str:
        """
        组装并填充指定名称的提示模板。
        支持嵌套模板，通过关键字参数传递子提示的填充内容。
        """
        prompt_template = cls.get_prompt(name)
        return prompt_template.format(**kwargs)

    @classmethod
    def list_prompts(cls):
        """
        列出所有已注册的提示名称。
        """
        return list(cls._registry.keys())

registry = PromptsRegistry()

@PromptsRegistry.register('CLEAN_PROMPT')
def clean_prompt():
    # 用于判断chunk是否useful
    return """Given the following chunk of text from an academic paper, please classify if the text is useful or not. Output 'Yes' for useful chunks and 'No' for useless chunks.
The following are some general traits of useful and useless chunks, along with some examples.
Useful chunks usually:
1. Mainly contain coherent English sentences.
2. Include one of the following: in-depth discussion scientific entities, coherent experiment procedures, meaningful comparison, intensive reasoning.

Useless chunks usually:
1. Are too short (only one or two sentences).
2. Contain non-relevant information to the main text such as title, author information, figure captions, references, declarations, etc.
3. Contain simple introduction to concepts without further discussions.
4. Contain ill-formatted formulae or tables that are not readable by humans.
5. Simply recorded the authors' experimental procedures without explicit order.
Examples of useful chunks:
{example_useful}
Examples of useless chunks:
{example_useless}
Text to classify: {chunk}
usefulness: Yes or No

Format instructions:
{format_instructions}
"""

@PromptsRegistry.register('EXAMPLES_USEFUL')
def examples_useful():
    # useful chunk示例
    return """ 'd was accurate according to our criteria (0.8 < K d / K d,inp < 1.25) for r 1 / r 2 2.5 but not for r 1 / r 2 5. At r 1 / r 2 = 0.25 we obtained a binding isotherm with anomalous shape (Figures S2) and K d / K d,inp = 1.27. This anomaly was due to a numerical artifact from meshing in COMSOL; by using a more refined mesh we obtained K d / K d,inp < 1.02. The improvement in accuracy by mesh refinement may suggest that the large deviations in K d at r 1 / r 2 5 are also due to too coarse meshes as well. Thus, more refined and optimized meshes (in particular, for boundary regions between small and large areas) could improve K d determination in a virtual ACTIS experiment.

We confirmed this for the extreme value of r 1 / r 2 = 50 and found an optimal K d / K d,inp = 1.00 at the expense of excessively increasing the computational time (72 h instead of 3 h) and the potential risk of overfitting (SI). In order to keep studies consistent, comparable and in a reasonable time', 
"""

@PromptsRegistry.register('EXAMPLES_USELESS')
def examples_useless():
    # useless chunk示例
    return """ ' ASSOCIATED CONTENT Supporting InformationThe Supporting Information is available free of charge on the ACS Publications website and on ChemRxiv (DOI:10.26434/chemrxiv.12345644). Theoretical background for computer simulation and data evaluation; Simulation of separagrams; Figure , Variation in k off, inp-separagrams and binding isotherms; Figure , Variation in injection loop dimensions -separagrams and binding isotherms; Figure , Variation in injection loop dimensions -sample-plug distribution; Figure , Variation in separation capillary radii -separagrams and binding isotherms; Figure , Velocity streamlines at different separation capillary radii; Figure , Variation in the initial', 
"""

@PromptsRegistry.register('REASONING_PROMPT')
def reasoning_prompt():
    # 判断当前chunk适合从哪个角度出题
    return """Please identify all the suitable types of questions to generate given a piece of text. Your available options are: ['Procedural', 'Comparative', 'Causal', 'Conditional', 'Evaluative', 'Predictive', 'Explanatory']. Please choose solely from the options. The options are defined as follows:
A Procedural question asks about the order between steps in a clearly formulated procedure. These procedures are often indicated by words such as 'first', 'then', 'finally', followed by actions.
A Comparative question asks about the relation between mutual properties of comparable entities, Common mutual properties include numbers, years, etc.
A Causal question asks about the reasons for a specific phenomenon. The phenomenon can be given implicitly or by explicit clauses such as 'for example'.
A Conditional question asks about the possible outcomes given a scenario. Scenarios are often given by conditional clauses such as 'if', 'when', etc.
An Evaluative question asks about the benefits and drawbacks of a given entity.
A Predictive question asks for reasonable inference, often on the properties of entities closely related to but not mentioned in the text.
An Explanatory question asks for a component from a statement made in the text.
Text: {text}
Structure your output in the following format:
Process: <Record here in detail how you go through each step of the instruction.>
Reasoning_types: <The reasoning types you chose>
Format instructions:
{format_instructions}
"""



@PromptsRegistry.register('PROCEDURAL_PROMPT')
def procedural_prompt():
    return """Please follow the instruction below to formulate a Procedural question based on the given text. A Procedural question asks about the order between steps in a clearly formulated procedure. These procedures are often indicated by words such as 'first', 'then', 'finally', followed by actions. You should go through the entire text and form questions only based on complete sentences.
1. Identify the procedure mentioned in the text. If no processes are mentioned, skip the following steps and output 'NaN' for <question>, <answer> and <context>.
2. List all steps in the process mentioned by the question in the exact same order as provided.
3. Choose one step (step1) from the process.
4. Determine its position in the process. i.e., where is it ranked in the process, the first, the second, or other?
5. Raise a question in the format: What is the <position> step in <summary of the process>?
6. Optionally, choose another step (step2) from the process. Determine the relative position of step1 to step2.
7. Raise a question in the following format: What is the <ordinal, relative position> step before/after <step2> in <summary of the process>? Replace the original question with the new one.
8. Record the question, answer, and context in the output. <question> should be the question you raised. <answer> should be step1, rephrased to be grammatically correct when necessary. <context> should be the original text containing the full process only.
Text: {text}
Structure your output in the following format:
Process: <Record here in detail how you go through each step of the instruction.>
Question: <question>
Answer: <answer>
Context: <context>
Format instructions:
{format_instructions}
"""


@PromptsRegistry.register('COMPARATIVE_PROMPT')
def comparative_prompt():
    return """Please follow the instruction below to formulate a Comparative question based on the given text. A Comparative question asks about the relation between mutual properties of comparable entities, Common mutual properties include numbers, years, etc. You should go through the entire text and form questions only base on complete sentences.
 1. Identify the comparable entities in the text, the comparable properties, e.g. numbers, years, etc, and their relation from the text. If there are no comparable 
properties or no relations are mentioned, skip the following steps and output 'NaN' for <question>, <answer> and <context>. 
 2. Identify the entities associated with the comparable values. 
 3. Randomly choose at least two entities and raise a question which asks about the relation between the comparable values of these entities. You shoule not disclose information on the relation in the question.
 4. Record the question, answer, and context in the output. <question> should be the question you raised. <answer> should be the relation you are asking for, including the result of comparison (e.g. bigger, smaller, similar, etc). Rephrase the answer to be grammatically correct. <context> should be all sentences in the original text excerpts describing the entities and their comparable values only. 
 Text: {text} Structure your output in the following format: Process: <Record here in detail how you go though each step of the instruction.> Question: <question> Answer: <answer> Context: <context> Format instructions: 
{format_instructions}
"""

@PromptsRegistry.register('CAUSAL_PROMPT')
def causal_prompt():
    return """
Please follow the instruction below to formulate a Causal question based on the given text. A Causal question asks about the reasons for a specific phenomenon. The phenomenon can be given implicitly or by explicit clauses such as 'for example'. You should go through the entire text and form questions only base on complete sentences.
 1. Identify the reasoning and scenario in the text. If no examples are mentioned, skip the following steps and output 'NaN' for <question>, <answer> and <context>. 2. Rephrase the scenario into a question. Do not add or delete any information. 
 3. Record the question, answer, and context in the output. <question> should be the question you raised. <answer> should be an explanation of the scenario based on the reasoning, rephrased to be grammatically correct when necessary. <context> should be all sentences in the original text containing the claims only. Text: {text} Structure your output in the following format: Process: <Record here in detail how you go though each step of the instruction.> Question: <question> Answer: <answer> Context: <context>
Format instructions: 
{format_instructions}
"""

@PromptsRegistry.register('CONDITIONAL_PROMPT')
def conditional_prompt():
    return """Please follow the instruction below to formulate a Conditional question based on the given text. A Conditional question asks about the possible outcomes given a scenario. Scenarios are often given by conditional clauses such as 'if', 'when', etc. You should go through the entire text and form questions only base on complete sentences.
 1. Identify the text containing conditions, e.g. clauses with 'if'. If no conditions are mentioned, skip the following steps and output 'NaN' for <question>, <answer> and <context>. 2. Identify the possible scenarios and the corresponding actions. 
 3. Formulate a question which asks for the action given one of the scenarios. You can choose scenarios not mentioned in the text. 
 4. Record the question, answer, and context in the output. <question> should be the question you raised. <answer> should be the corresponding action, rephrased to be grammatically correct when necessary. <context> should be all sentences in the original text containing the statements only. 
 Text: {text} Structure your output in the following format: Process: <Record here in detail how you go though each step of the instruction.> Question: <question> Answer: <answer> Context: <context> Format instructions: 
{format_instructions}
"""

@PromptsRegistry.register('EVALUATIVE_PROMPT')
def evaluative_prompt():
    return """Please follow the instruction below to formulate an Evaluative question based on the given text. An Evaluative question asks about the benefits and drawbacks of a given entity. 
 You should go through the entire text and form questions only base on complete sentences.
 1. List all statements made in the text. Find if any statements explain the properties of a specific entity and imply value judgements. Define these statement as 'necessary statements'. If no statements satisfy the requirements, skip the following steps and output 'NaN' for <question>, <answer> and <context>. 2. Reformulate the 'necessary statements' in the format: <entity>: <properties> 
 3. Classify the properties as positive or negative. 

4. Raise a question based on the format: What are the pros and cons / benefits / drawbacks of <entity>? Paraphrase the question. 
 5. Record the question, answer, and context in the output. <question> should be the question you raised. <answer> should contain all <properties> associatd with the authors' attitude, rephrased to be grammatically correct when necessary. <context> should be all sentences in the original text containing 'necessary statements'only. Text: {text} Structure your output in the following format: Process: <Record here in detail how you go though each step of the instruction.> Question: <question> Answer: <answer> Context: <context> Format instructions: 
{format_instructions}
"""

@PromptsRegistry.register('PREDICTIVE_PROMPT')
def predictive_prompt():
    return """Please follow the instruction below to formulate a Predictive question based on the given text. A Predictive question asks for reasonable inference, often on the properties of entites closely related to but not mentioned in the text. You should go through the entire text and form questions only base on complete sentences.
 1. List all statements made in the text. Find if any statements explain the properties of a specific entity. Define these statement as 'necessary statements'. If no statements satisfy the requirements, skip the following steps and output 'NaN' for <question>, <answer> and <context>. 2. Randomly choose from the following one category of transformations with equal probability: 
 a. Negation 
 b. Generalization/specification 
 c. Analogy 
 3. Apply to the most suitable entity-property pair. The transformed entity and property must both make sense scientifically. 
 4. Raise a question which asks for the property of the transformed entity. Do not disclose any information about the transformed property in the question. 
 5. Record the question, answer, and context in the output. <question> should be the question you raised. <answer> should contain <transformed properties> and be rephrased to be grammatically correct when necessary. <context> should be all s entences in the original text containing the 'necessary statements' only. 
 Text: {text}
Structure your output in the following format: Process: <Record here in detail how you go though each step of the instruction.> Question: <question> Answer: <answer> Context: <context> Format instructions: 
{format_instructions}
"""

@PromptsRegistry.register('EXPLANATORY_PROMPT')
def explanatory_prompt():
    return """Please follow the instruction below to formulate an Explanatory question based on the given text. An Explanatory question asks for a component from a statement made in the text. 
 You should go through the entire text and form questions only base on complete sentences.
 1. List all statements made in the text. 2. Choose a statement and replace part of it with an appropriate interrogative pronoun. The part you replace should be specific. You should not mention the replaced information in the question. 
 3. Rephrase the question to be grammatically correct. 
 4. Record the question, answer, and context in the output. <question> should be the question you raised. <answer> should be the part you replaced, rephrased to be grammatically correct when necessary. <context> should be all sentences in the original text containing the chosen statement only. 
 Text: {text} Structure your output in the following format: Process: <Record here in detail how you go though each step of the instruction.> Question: <question> Answer: <answer> Context: <context> Format instructions: 
{format_instructions}
"""


@PromptsRegistry.register('DIFFICULTY_PROMPT')
def difficulty_prompt():
    return """You are given a text chunk and a question-answer pair derived from the chunk. Please assign one of the labels from 'Easy', 'Medium' and 'Hard' to <difficulty>, where the easiest question is one whose answer is directly available in a single sentence in the chunk, and the hardest question is one which requires information from multiple sentences in the chunk and complex reasoning to arrive at the answer. Question: {question} Answer: {answer} 
Chunk: {chunk} Structure your output in the following format: Difficulty: <difficulty> Format instructions: 
{format_instructions}
"""


@PromptsRegistry.register('CLEAN_PROMPT_zh')
def clean_prompt_zh():
    # 用于判断chunk是否有用
    return """给定以下来自教材的文本片段，请判断该文本片段是否可以用于教师出题参考。如果是有用的文本片段，请输出'是'；如果是无用的文本片段，请输出'否'。
# 判断标准：
有用的文本片段通常：
1. 包含连贯的句子，结构清晰。
2. 包含以下内容之一： 深入讨论的科学概念、实体或现象。详细描述的实验过程或方法。有意义的比较分析或结果讨论。复杂的推理过程或理论解释。

无用的文本片段通常：
1. 太短（只有一两句）。
2. 包含与正文无关的信息，如标题、作者信息、图表说明、参考文献、版权声明等。
3. 仅简单介绍概念而没有进一步的讨论或解释。
4. 包含无法被人类读取的格式错误的公式或表格。
5. 仅记录了实验步骤或过程，缺乏逻辑顺序或详细说明。
有用的文本片段示例：
{example_useful}
回答：是，因为该片段详细描述了实验结果及其解释，包含科学概念和理论分析。
无用的文本片段示例：
{example_useless}
回答：否，因为该片段仅包含版权信息，介绍了计算机科学导论的内容和作者信息，与教材正文内容无关。
需要分类的文本：{chunk}
文本有用性：是 或 否

格式说明：
{format_instructions}
"""

@PromptsRegistry.register('EXAMPLES_USEFUL_zh')
def examples_useful_zh():
    # 有用的文本片段示例
    return """'根据我们的实验结果，当温度升高时，反应速率显著增加。这一现象可以通过阿伦尼乌斯方程来解释，说明温度对分子碰撞频率和能量的影响。此外，不同催化剂在相同条件下表现出不同的活性，这表明催化剂的性质对反应机制有重要影响。在本研究中，我们采用了双盲对照实验设计，以减少偏差的影响。参与者被随机分配到实验组和对照组，确保两组在初始条件上的可比性。数据通过SPSS软件进行统计分析，结果显示实验组在关键指标上有显著提升（p < 0.05）。
"""


@PromptsRegistry.register('EXAMPLES_USELESS_zh')
def examples_useless_zh():
    # 无用的文本片段示例
    return """'Foundations of Computer Science D Third Edition  
机械工业出版社China Machine Press  
计算机科学导论 原书第3版Foundations of Computer Seience Thrd Edrtion三遗是一本条理清晰并姐  
"""


@PromptsRegistry.register('REASONING_PROMPT_zh')
def reasoning_prompt_zh():
    # 判断当前chunk适合从哪个角度出题
    return """请根据以下文本，确定适合生成的所有问题类型。可选的问题类型包括：
- **程序性**：涉及按照特定步骤或顺序完成任务的问题。
- **比较性**：涉及对两个或多个概念、过程或实体进行对比的问题。
- **因果性**：探讨某一现象或结果背后原因的问题。
- **条件性**：基于特定条件或情境下可能发生的情况提出的问题。
- **评估性**：评估某一概念、过程或实体的优缺点的问题。
- **预测性**：基于现有信息对未来情况进行推测的问题。
- **解释性**：要求详细说明某一概念、过程或现象的问题。
示例：
输入的文本：
3.2.1 存储整数
整数是完整的数字（即没有小数部分）。例如，134和-125是整数而134.23和-0.235则不是。整数可以被当作小数点位置固定的数字：小数点固定在最右边。因此，定点表示法用于存储整数，如图3-4所示。在这种表示法中，小数点是假定的，但并不存储。
图 3-4整数的定点表示法
但是，用户（或程序）可能将整数作为小数部分为0的实数存储。这是可能发生的，例如，整数太大以至于无法定义为整数来存储。为了更有效地利用计算机内存，无符号和有符号的整数在计算机中存储方式是不同的。
输出：
过程：首先，我分析了文本内容，识别出涉及整数存储的方法和概念。然后，我确定哪些问题类型可以有效地围绕这些内容展开。定点表示法和无符号/有符号整数的存储方式对比适合比较性问题；探讨为何需要不同的存储方式适合因果性问题；解释定点表示法的工作原理适合解释性问题。 
推理类型：比较性, 因果性, 解释性

请按示例的格式组织输出：
过程：<详细记录您如何一步步完成指令的过程。>
推理类型：<您选择的推理类型>

输入的文本：
{text}
"""

@PromptsRegistry.register('PROCEDURAL_PROMPT_zh')
def procedural_prompt_zh():
    return """请根据以下指令，根据给定的文本构造一个程序性问题。程序性问题询问明确制定的程序中步骤的顺序。这些程序通常通过“首先”、“然后”、“最后”等词语和相应的动作来表示。您应该遍历整个文本，仅基于完整的句子构造问题。
1. 确定文本中提到的程序。如果没有提到过程，请跳过以下步骤，并输出 'NaN' 作为 <question>、<answer> 和 <context>。
2. 按照提供的顺序列出过程中的所有步骤。
3. 从过程中选择一个步骤（步骤1）。
4. 确定该步骤在过程中的位置，即它在过程中的排名是第一个、第二个还是其他？
5. 提出一个问题，格式为：What is the <position> step in <summary of the process>?（<过程的总结> 中第 <位置> 步是什么？）
6. 可选：从过程中选择另一个步骤（步骤2）。确定步骤1与步骤2的相对位置。
7. 提出一个问题，格式为：What is the <ordinal, relative position> step before/after <step2> in <summary of the process>?（在 <过程的总结> 中，<步骤2> 之前/之后的第 <顺序号，位置> 步是什么？）将原问题替换为新问题。
8. 在输出中记录问题、答案和上下文。<question> 应该是您提出的问题。<answer> 应该是步骤1，必要时重新表述为语法正确的形式。<context> 应该是包含完整过程的原始文本。
请按以下格式组织输出：
过程：<详细记录您如何一步步完成指令的过程。>
上下文：<context>
问题：<question>
答案：<answer>

输入的文本：{text}
"""


@PromptsRegistry.register('COMPARATIVE_PROMPT_zh')
def comparative_prompt_zh():
    return """请根据以下指令，根据给定的文本构造一个比较性问题。比较性问题询问可比实体的共同属性之间的关系。常见的共同属性包括数字、年份等。您应该遍历整个文本，仅基于完整的句子构造问题。
1. 确定文本中的可比实体、共同属性（例如数字、年份等）及其关系。如果没有提到可比属性或关系，请跳过以下步骤，并输出 'NaN' 作为 <question>、<answer> 和 <context>。
2. 确定与可比值相关联的实体。
3. 随机选择至少两个实体，提出一个问题，询问这些实体的可比值之间的关系。问题中不应透露关于关系的任何信息。
4. 在输出中记录问题、答案和上下文。<question> 应该是您提出的问题。<answer> 应该是您询问的关系，包括比较结果（例如，更大、更小、相似等）。必要时重新表述答案为语法正确的形式。<context> 应该是描述实体及其可比值的原始文本中的所有句子。
请按以下格式组织输出：
过程：<详细记录您如何一步步完成指令的过程。>
上下文：<context>
问题：<question>
答案：<answer>

输入的文本：{text}
"""

@PromptsRegistry.register('CAUSAL_PROMPT_zh')
def causal_prompt_zh():
    return """
请根据以下指令，根据给定的文本构造一个因果性问题。因果性问题询问特定现象的原因。现象可以是隐式给出的，或者通过显式短语如“例如”来表达。您应该遍历整个文本，仅基于完整的句子构造问题。
1. 确定文本中的推理和情境。如果没有提到示例，请跳过以下步骤，并输出 'NaN' 作为 <question>、<answer> 和 <context>。
2. 将情境重新表述为一个问题。请勿添加或删除任何信息。
3. 在输出中记录问题、答案和上下文。<question> 应该是您提出的问题。<answer> 应该是根据推理对情境的解释，必要时重新表述答案为语法正确的形式。<context> 应该是原始文本中包含相关主张的所有句子。
请按以下格式组织输出：
过程：<详细记录您如何一步步完成指令的过程。>
问题：<question>
答案：<answer>
上下文：<context>

输入的文本：{text}
"""

@PromptsRegistry.register('CONDITIONAL_PROMPT_zh')
def conditional_prompt_zh():
    return """请根据以下指令，根据给定的文本构造一个条件性问题。条件性问题询问在特定情境下可能的结果。情境通常通过条件句如“如果”、“当”等给出。您应该遍历整个文本，仅基于完整的句子构造问题。
1. 确定包含条件的文本，例如包含“如果”的子句。如果没有提到条件，请跳过以下步骤，并输出 'NaN' 作为 <question>、<answer> 和 <context>。
2. 确定可能的情境及其对应的行动。
3. 构造一个问题，询问在某一情境下的行动。您可以选择文本中未提到的情境。
4. 在输出中记录问题、答案和上下文。<question> 应该是您提出的问题。<answer> 应该是相应的行动，必要时重新表述答案为语法正确的形式。<context> 应该是原始文本中包含相关陈述的所有句子。
请按以下格式组织输出：
过程：<详细记录您如何一步步完成指令的过程。>
问题：<question>
答案：<answer>
上下文：<context>

输入的文本：{text}
"""

@PromptsRegistry.register('EVALUATIVE_PROMPT_zh')
def evaluative_prompt_zh():
    return """请根据以下指令，根据给定的文本构造一个评估性问题。评估性问题询问某一实体的优点和缺点。
您应该遍历整个文本，仅基于完整的句子构造问题。
1. 列出文本中的所有陈述。查找是否有任何陈述解释了某一实体的属性并暗示价值判断。将这些陈述定义为“必要陈述”。如果没有满足条件的陈述，请跳过以下步骤，并输出 'NaN' 作为 <question>、<answer> 和 <context>。
2. 将“必要陈述”重新表述为格式：<entity>: <properties>
3. 将这些属性分类为正面或负面。
4. 提出一个问题，格式为：What are the pros and cons / benefits / drawbacks of <entity>?（<entity> 的优点和缺点是什么？）对问题进行同义改写。
5. 在输出中记录问题、答案和上下文。<question> 应该是您提出的问题。<answer> 应该包含所有与作者态度相关的 <properties>，必要时重新表述答案为语法正确的形式。<context> 应该是原始文本中仅包含“必要陈述”的所有句子。
请按以下格式组织输出：
过程：<详细记录您如何一步步完成指令的过程。>
问题：<question>
答案：<answer>
上下文：<context>

文本：{text}
"""

@PromptsRegistry.register('PREDICTIVE_PROMPT_zh')
def predictive_prompt_zh():
    return """请根据以下指令，根据给定的文本构造一个预测性问题。预测性问题要求合理推断，通常是关于与文本密切相关但未提及的实体的属性。您应该遍历整个文本，仅基于完整的句子构造问题。
1. 列出文本中的所有陈述。查找是否有任何陈述解释了某一实体的属性。将这些陈述定义为“必要陈述”。如果没有满足条件的陈述，请跳过以下步骤，并输出 'NaN' 作为 <question>、<answer> 和 <context>。
2. 从以下选项中随机选择一类转换，概率相等：
   a. 否定
   b. 概括/具体化
   c. 类比
3. 将选择的转换应用于最合适的实体-属性对。转换后的实体和属性必须在科学上是合理的。
4. 提出一个问题，询问转换后实体的属性。问题中不得透露转换后的属性信息。
5. 在输出中记录问题、答案和上下文。<question> 应该是您提出的问题。<answer> 应该包含转换后的属性，并根据需要重新表述为语法正确的形式。<context> 应该是原始文本中仅包含“必要陈述”的所有句子。
请按以下格式组织输出：
过程：<详细记录您如何一步步完成指令的过程。>
问题：<question>
答案：<answer>
上下文：<context>

文本：{text}
"""

@PromptsRegistry.register('EXPLANATORY_PROMPT_zh')
def explanatory_prompt_zh():
    return """请根据以下指令，根据给定的文本构造一个解释性问题。解释性问题询问文本中陈述的某一部分内容。
您应该遍历整个文本，仅基于完整的句子构造问题。
1. 列出文本中的所有陈述。
2. 选择一个陈述，并用合适的疑问代词替换其中的一部分。您替换的部分应该是具体的。问题中不应提及被替换的信息。
3. 将问题重新表述为语法正确的形式。
4. 在输出中记录问题、答案和上下文。<question> 应该是您提出的问题。<answer> 应该是您替换的部分，必要时重新表述为语法正确的形式。<context> 应该是原始文本中包含所选陈述的所有句子。
文本：{text}
请按以下格式组织输出：
过程：<详细记录您如何一步步完成指令的过程。>
问题：<question>
答案：<answer>
上下文：<context>
"""

@PromptsRegistry.register('DIFFICULTY_PROMPT_zh')
def difficulty_prompt_zh():
    return """请根据给定的文本、问题及其答案，对问题的难度进行评估。请将难度标签设定为以下三种之一：'Easy'（简单）、'Medium'（中等）和 'Hard'（困难）。每个标签的定义如下：
- **Easy（简单）**：问题的答案可以直接从文本中的单一句子中找到，无需额外的推理或信息整合。
- **Medium（中等）**：问题的答案需要从文本中的多个句子中提取信息，可能需要一定程度的推理。
- **Hard（困难）**：问题的答案需要从文本中的多个句子中提取信息，并通过复杂的推理或综合分析才能得出。
{
    "难度": "<difficulty>"
}
示例输入：

问题：太阳系中最大的行星是什么？
答案：木星
文本：太阳系由八大行星组成，其中木星是最大的行星，其直径约为142,984公里。
示例输出：
{
    "难度": "Easy"
}
输入：
{text}
"""

@PromptsRegistry.register('EVALUATION_PROMPT_zh')
def evaluation_prompt_zh():
    return """请根据以下生成的题目，从可读性、适切性、复杂性和参与度四个维度对其质量进行评估。每个维度的评分范围为0到5分，并请具体说明每个分数对应的程度。最后以JSON格式输出评估结果，格式如下：
{{
    "可读性": {{
        "得分": 分数,
        "说明": "具体说明"
    }},
    "适切性": {{
        "得分": 分数,
        "说明": "具体说明"
    }},
    "复杂性": {{
        "得分": 分数,
        "说明": "具体说明"
    }},
}}

以下是评分维度的详细说明：

1. **可读性**：评估题目是否易于阅读和理解。
   - 0分：题目非常难以理解，存在严重的语法或用词错误。
   - 1分：题目理解起来非常困难，表达不清晰。
   - 2分：题目有一定难度，部分学生可能难以理解。
   - 3分：题目基本清晰，大多数学生能够理解。
   - 4分：题目表达清晰，易于理解，适合大多数学生。
   - 5分：题目极其清晰，语言简洁明了，非常易于理解。

2. **学科适宜性**：评估题目在语义上是否与相应学科对齐，符合教学目标。
   - 0分：题目与学科内容完全不相关。
   - 1分：题目与学科内容关联极弱，几乎没有相关性。
   - 2分：题目与学科内容关联较弱，相关性不强。
   - 3分：题目与学科内容基本相关，符合教学目标。
   - 4分：题目与学科内容高度相关，完全符合教学目标。
   - 5分：题目与学科内容完美对齐，极大地支持教学目标。

3. **复杂性**：评估题目需要的推理或认知努力程度。
   - 0分：题目完全不具备认知挑战，过于简单。
   - 1分：题目几乎没有认知挑战，难度极低。
   - 2分：题目有轻微的认知挑战，难度较低。
   - 3分：题目具备适度的认知挑战，难度适中。
   - 4分：题目具有较高的认知挑战，难度较大。
   - 5分：题目极具认知挑战，难度很高，适合高水平学生。

请根据以上维度和说明，对以下生成的题目进行评估，并按照指定的JSON格式输出结果。
{text}
"""


@PromptsRegistry.register('LABEL_CHUNK_PROMPT_zh')
def label_chunk_prompt_zh():
    return """请从以下文本片段中提取三个标签：类别（category）、子类别（subcategory） 和 知识点（knowledge point）。这些标签应准确反映文本内容的主题和主要知识点。
"category": "类别名称",
"subcategory": "子类别名称",
"knowledge_point": "知识点描述"
示例：
- 文本：
应该使用位模式，它是一个序列，有时也被称为位流。图3-2展示了由16个位组成的位模式。它是0和1的组合。这就意味着，如果我们需要存储一个由16个位组成的位模式，那么需要16个电子开关。如果我们10001010111111需要存储1000个位模式，每个16位，那么需要16000个开关。 图3-2位模式通常长度为8的位模式被称为1字节。有时用字这个术语指代更长的位模式。  
正如图3-3所示，属于不同数据类型的数据可以以同样的模式存储于内存中。  
图 3-3不同数据类型的存储  
如果使用文本编辑器（文字处理器），键盘上的字符A可以以8位模式01000001存储。如果使用数学程序，.同样的8位模式可以表示数字65。类似地，同样的位模式可表示部分图像、部分歌曲、影片中的部分场景。计算机内存存储所有这些而无需辨别它们表示的是何种数据类型。  
# 3.数据压缩  
为占用较少的内存空间，数据在存储到计算机之前通常被压缩。数据压缩是一个广阔的主题，所以我们用整个第15章来讲述。  
# 数据压缩在第15章讨论。  
# 4.错误检测和纠正  
另一个与数据有关的话题是在传输和存储数据时的错误检测和纠正。我们在附录H中简要讨论这个话题。  
【错得检测和纠正在附景中  
# 3.2存储数字  
在存储到计算机内存中之前，数字被转换到二进制系统，如第2章所述。但是，这里还  
有两个问题需要解决：  
1）如何存储数字的符号。  
2）如何显示十进制小数点。  
有多种方法可处理符号问题，本章后面陆续讨论。对于小数点，计算机使用两种不同的表示方法：定点和浮点。第一种用于把数字作为整数存储——没有小数部分，第二种把数字作为实数存储一带有小数部分。  
# 3.2.1 存储整数  
整数是完整的数字（即没有小数部分）。例如，134和-125是整数而134.23和-0.235则不是。整数可以被当作小数点位置固定的数字：小数点固定在最右边。因此，定点表示法用于存储整数，如图3-4所示。在这种表示法中，小数点是假定的，但并不存储。  
图 3-4整数的定点表示法  
但是，用户（或程序）可能将整数作为小数部分为0的实数存储。这是可能发生的，例如，整数太大以至于无法定义为整数来存储。为了更有效地利用计算机内存，无符号和有符号的整数在计算机中存储方式是不同的。  
蓝效通使用定点表示法存储在内存中  
# 1.无符号表示法  
无符号整数是只包括零和正数的非负整数。它的范围介于0到无穷大之间。然而，由于计算机不可能表示这个范围的所有整数，通常，计算机都定义了一个常量，称为最大无符号整数，它的值是 $(2^{{n}}\!\!-\!1\ )$ 。这里 $\pmb{{n}}.$ 就是计算机中分配用于表示无符号整数的二进制位数。  
- 输出：
"category": "计算机科学",
"subcategory": "数据存储与表示",
"knowledge_point": "位模式、字节结构、定点与浮点表示、无符号与有符号整数存储"
- 需要你判断的文本：
{chunk}
"""


if __name__ == "__main__":
    # 测试
    example_useful = PromptsRegistry.assemble('EXAMPLES_USEFUL')
    example_useless = PromptsRegistry.assemble('EXAMPLES_USELESS')
    format_instructions = "参考示例只输出是否，以及原因"

    # 组装 CLEAN_PROMPT
    clean_prompt = PromptsRegistry.assemble(
        'CLEAN_PROMPT',
        example_useful=example_useful,
        example_useless=example_useless,
        chunk="This is a sample text chunk to classify.",
        format_instructions=format_instructions
    )

    print("=== CLEAN_PROMPT ===")
    print(clean_prompt)

    # 列出所有注册的提示
    print("\n=== Registered Prompts ===")
    for prompt_name in PromptsRegistry.list_prompts():
        print(prompt_name)
