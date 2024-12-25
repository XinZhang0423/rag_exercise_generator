import re
import os
import json
import time
from numpy import isin
from tqdm import tqdm
from api_utils import APIClient

def check_legal_simple(question_type,question,choices):
    client=APIClient()
    choices='' if choices is None else choices
    prompt="""
    以下是解析出来的内容，因此会有截断和不全的情况，请判断以下内容是否构成一个完整且合理的题目。如果构成，回答"True"，否则回答"False"，同时需要输出判断的理由
    以下是一些示例：
    例1：
    类型：freeform
    问题：解释为什么计算机不能解决那些计算机外部世界无解决方法的问题。
    选项：null
    答案：True
    理由：这是一个完整且合理的题目，符合freeform类型的问题要求，询问了计算机无法解决的某些问题的原因。

    例2：
    类型：freeform
    问题：（）小程序
    选项：null
    答案：False
    理由：题目不完整，括号中的部分没有具体内容，无法理解问题的含义，因此不能算作一个合理的问题。

    例3：
    类型：multiple_choice
    问题：以下哪个是正确的？
    选项：A. 选项A B. 选项B C. 选项C
    答案：False
    理由：没有具体的选项内容，且"以下哪个是正确的"这个问题也不全
    
    例4:
    类型：freeform
    问题：这一部分包括难度更大的题目，这些题目的求解需要对该章讨论的内容有更深层次的理解。我强烈推荐学生去尝试求解这部分的全部题目。奇数编号练习题的答案也已经公布在了本书网站以便学生进行核对。
    选项：null
    答案：False
    理由：这段话更像是指导性建议，而不是具体的题目。因此不构成一个完整的题目。

    现在请判断以下内容：
    类型：{0}
    问题：{1}
    选项：{2}
    """
    messages=[{'role':'user','content':prompt.format(question_type,question,choices)}]
    sampling_params={'temperature':0,'top_p':0.8}
    response=client.call_zhipu(model='GLM-4-flash',messages=messages,sampling_params=sampling_params)
    if isinstance(response,str):
        if 'True' in response:
            return True,response.split('理由：')[1]
        else:
            return False,response.split('理由：')[1]
    else:
        return False,response
        
def remove_images(text):
    """
    用re规则去除OCR文本中的图像数据（以 ![] 开头的图像链接），并返回统计信息。
    """
    original_char_count = len(text.split())  # 统计原始字数

    # 使用 finditer 查找所有匹配的图片链接，并统计数量
    image_matches = list(re.finditer(r'!\[.*?\]\(.*?\)', text))
    images_removed_count = len(image_matches)

    # 使用 sub 替换所有匹配的图片链接为空字符串
    cleaned_text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    cleaned_char_count = len(cleaned_text.split())  # 统计去除图片后的字符数

    return cleaned_text, original_char_count, images_removed_count, cleaned_char_count


def extract_exercises(text, section_title):
    """
    提取OCR文本中的习题数据
    """
    title_matches = [m.start() for m in re.finditer(rf'({section_title})\s*$', text, re.MULTILINE)]
    all_exercises = []

    for i in range(len(title_matches)):
        start = title_matches[i]
        end = title_matches[i + 1] if i + 1 < len(title_matches) else len(text)

        exercises_text = text[start:end]

        # 去除标题行本身
        exercises_text = exercises_text.split('\n', 1)[1] if '\n' in exercises_text else ""

        # 在这里添加截断逻辑：查找下一个标题
        next_title_match = re.search(r'#\s', exercises_text)  # 关键修改：简化
        if next_title_match:
            exercises_text = exercises_text[:next_title_match.start()]

        exercises_text = exercises_text.strip()
        exercises = [line.strip() for line in exercises_text.split("\n") if line.strip()]
        all_exercises.extend(exercises)

    return all_exercises

def clean_markdown(text,chunk_size=2000):
    n=len(text)
    chunks=[]
    res=''
    for chunk_index in range(0,n,chunk_size):
        chunk=text[chunk_index:chunk_index+chunk_size+1]
        chunks.append(chunk)
    client=APIClient()
    sampling_params={'temperature':0,'top_p':0.8}
    prompt="""你是一个教材修订者，以下是ocr识别的数理统计教材，请检查并修正以下来自于数理统计教材中的所有错误，包括但不限于 OCR 识别错误，公式错误和格式错误。如果没有错误，要严格保持不变，保留原始内容的结构和格式。你不要说其他内容，只输出修改后的结果！
    {0}
    """
    for chunk in tqdm(chunks):
        messages=[{'role':'user','content':prompt.format(chunk)}]
        response=client.call_zhipu(model='GLM-4-flash',messages=messages,sampling_params=sampling_params)
        if isinstance(response,str):
            res+=response
        else:
            try:
                print(response)
            except Exception as e:
                print(e)
        # break
    # print(res)
    return res

def extract_exercises_llm(text):
    title_matches = [m.start() for m in re.finditer(rf'({"习题"})\s*$', text, re.MULTILINE)]
    chunks = []

    for i in range(len(title_matches)):
        start = title_matches[i]
        end = title_matches[i + 1] if i + 1 < len(title_matches) else len(text)

        exercises_text = text[start:end]

        # 去除标题行本身
        exercises_text = exercises_text.split('\n', 1)[1] if '\n' in exercises_text else ""

        # 在这里添加截断逻辑：查找下一个标题
        next_title_match = re.search(r'#\s', exercises_text)  # 关键修改：简化
        if next_title_match:
            exercises_text = exercises_text[:next_title_match.start()]

        exercises_text = exercises_text.strip()
        chunks.append(exercises_text)
    
    prompt="""我给你一段包含多道习题的文本，需要你逐个解析出其中的题目，你需要去除它的题号等无用信息，同时判断这是否是一个完整的题目，即是否存在公式异常，题目缺少条件等，如果是则保留，如果不是则跳过。你只需要输出最后的json输出，输出格式为：[{{'question_type':'freeform','question':question1}},{{'question_type':'freeform','question':question2}}] 
    ##待提取的文本
    {0}
    """
    client=APIClient()
    sampling_params={'temperature':0,'top_p':0.8}
    res=''
    for chunk in tqdm(chunks):
        messages=[{'role':'user','content':prompt.format(chunk)}]
        response=client.call_openai(model='gpt-4o-mini',messages=messages,sampling_params=sampling_params)
        if isinstance(response,str):
            res+=response
        else:
            try:
                print(response)
            except Exception as e:
                print(e)
        # break
    # print(res)
    return res
    

def clean_exercises_llm(question):
    prompt="""**背景：**  
    你将收到一段数理统计的习题文本。需要你判断该文本是否构成一个完整的题目，并检查其中的公式是否正确，是否缺少必要的条件。你的任务是根据题目内容和公式的完整性给出适当的判断。

    **任务目的：**  
    判断给定的文本是否为有效的数理统计题目，并对其中公式的正确性进行评估。如果题目完整且公式正确，则返回原文本；如果公式存在轻微问题或缺少部分条件，则返回修改后的版本,比如给定分布求另一个分布，你可以依据情况对公式进行补全；如果题目严重缺失条件，或者文本本身不完整（如乱码），则返回 `False` 和空字符串。

    **步骤：**  
    1. 阅读给定的文本，判断其是否为一个完整且有效的数理统计题目。
    2. 检查其中的公式是否符合规范，是否存在缺失条件或符号错误。
    3. 根据检查结果进行判断：
    - 如果题目完整且公式正确，返回 `True` 和原文本。
    - 如果公式有轻微问题或缺少条件，但整体有效，返回 `True` 和修改后的版本。
    - 如果题目有严重缺失或是乱码，返回 `False` 和空字符串。

    **输出格式：**  
    - 如果题目和公式正确：`True, <原始文本>`
    - 如果题目正常但公式有轻微问题或缺少条件：`True, <修改后的文本>`
    - 如果题目严重缺失或乱码：`False, ''`
    - 只需要输出结果，无需给出理由和分析
    **示例：**  
    1. **输入：**  
    写出下列随机试验的样本空间S：记录一个班一次数学考试的平均分数（设以百分制记分）。  
    **输出：**  
    `True, 写出下列随机试验的样本空间S：记录一个班一次数学考试的平均分数（设以百分制记分）。`
    
    2. **输入：**  
    $\\hat{{y}}=774.\\,0125\,{{-}}\,0.\\,35915x$  
    **输出：**  
    `False, ''`
    ##输入：
    {0}
    """
    
    client=APIClient()
    sampling_params={'temperature':0,'top_p':0.8}
    messages=[{'role':'user','content':prompt.format(question)}]
    response=client.call_openai(model='gpt-4o-mini',messages=messages,sampling_params=sampling_params)
    if isinstance(response,str):
        return response
    else:
        try:
            print(response)
        except Exception as e:
            print(e)
        return response

if __name__=="__main__":
    
    # 示例文本
    intro_text = """
    ![](images/fb82577df7dbcb65eb30648e79a52a64d673601553d91f1cd7db3c891833fe46.jpg)  
    图 2-1在十进制系统中使用位置量表示整数
    """
    text = """
    # 复习题  
    1.定义一个基于图灵模型的计算机。  
    2.定义一个基于冯·诺依曼模型的计算机。  
    3.在基于图灵模型的计算机中，程序的作用是什么？  
    4.在基于冯·诺依曼模型的计算机中，程序的作用是什么？  

    # 练习题  
    1.解释为什么计算机不能解决那些计算机外部世界无解决方法的问题。  
    2.如果一台小的便宜的计算机可以做大型昂贵的计算机同样能做的事情，为什么人们需要大的呢？  
    3.研究Pascaline计算器，看看它是否符合图灵模型。  
    4.研究莱布尼茨之轮（Leibnitz'sWheel)，看看它是否符合图灵模型。  
    5.研究雅卡尔提花织机（Jacquardloom），看看它是否符合图灵模型。  
    6.研究查尔斯·巴比奇分析引擎，看看它是否符合冯·诺依曼模型。  
    7.研究ABC计算机，看看它是否符合冯·诺依曼模型。  
    8.研究并找出键盘起源于哪一代计算机。
    
    # 复习题  
    1.定义一个基于图灵模型的计算机。  
    2.定义一个基于冯·诺依曼模型的计算机。  
    3.在基于图灵模型的计算机中，程序的作用是什么？  
    4.在基于冯·诺依曼模型的计算机中，程序的作用是什么？ 
    """

    # intro
    # file_path="/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro/intro.md"
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     text=f.read()
    
    # # 去除图片信息
    # cleaned_text, original_text_count, images_removed_count, cleaned_text_count=remove_images(text)
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro/intro_image_free.md",'w',encoding='utf-8') as f:
    #     f.write(cleaned_text)
    # image_info={"original_text_count":original_text_count,"images_removed_count":images_removed_count,"cleaned_text_count":cleaned_text_count,"word_removed":original_text_count-cleaned_text_count}
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro/intro_image_info.json",'w',encoding='utf-8') as f:
    #     f.write(json.dumps(image_info,ensure_ascii=False))
        
        
    # # 提取题目
    # # 复习题
    # exercises_1=extract_exercises(cleaned_text,"复习题")
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro/intro_exercises_1.jsonl",'w',encoding='utf-8') as f:
    #     for i,exe in enumerate(exercises_1):
    #         if exe.startswith('a'):
    #             continue
    #         cleaned_exe=''.join([s for s in exe if not s.isdigit() and s!='.'])
    #         if i<len(exercises_1)-1 and exercises_1[i+1].startswith('a'):
    #             print(i,'是合适的题目')
    #             f.write(json.dumps({'id':i,'question_type':'multichoice','question':cleaned_exe,'choices':exercises_1[i+1]},ensure_ascii=False)+'\n')
    #         else:
    #             f.write(json.dumps({'id':i,'question_type':'freeform','question':cleaned_exe,'choices':None},ensure_ascii=False)+'\n')
            
    # exercises_2=extract_exercises(cleaned_text,"练习题")
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro/intro_exercises_2.jsonl",'w',encoding='utf-8') as f:
    #     for i,exe in tqdm(enumerate(exercises_2)):
    #         if exe.startswith('a'):
    #             continue
    #         cleaned_exe=''.join([s for s in exe if not s.isdigit() and s!='.'])
    #         is_legal,reason=check_legal_simple(question_type="freeform",question=cleaned_exe,choices=None)
    #         time.sleep(1)
    #         if is_legal:
    #             if i<len(exercises_1)-1 and exercises_1[i+1].startswith('a'):
    #                 f.write(json.dumps({'id':i,'question_type':'multichoice','question':cleaned_exe,'choices':exercises_1[i+1]},ensure_ascii=False)+'\n')
    #             else:
    #                 f.write(json.dumps({'id':i,'question_type':'freeform','question':cleaned_exe,'choices':None},ensure_ascii=False)+'\n')
    #         else:
    #             print(reason)
    
    # statistic
    file_path="/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic.md"
    with open(file_path, 'r', encoding='utf-8') as f:
        text=f.read()
    
    # 去除图片信息
    cleaned_text, original_text_count, images_removed_count, cleaned_text_count=remove_images(text)
    with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_image_free.md",'w',encoding='utf-8') as f:
        f.write(cleaned_text)
    image_info={"original_text_count":original_text_count,"images_removed_count":images_removed_count,"cleaned_text_count":cleaned_text_count,"word_removed":original_text_count-cleaned_text_count}
    with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_image_info.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(image_info,ensure_ascii=False))
    # res=clean_markdown(cleaned_text)
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_cleaned.md",'w',encoding='utf-8') as f:
    #     f.write(res)
        
    # 提取题目
    # 习题
    # exercises_1=extract_exercises(cleaned_text,"习题")
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_exercises_1.jsonl",'w',encoding='utf-8') as f:
    #     for i,exe in enumerate(exercises_1):
    #         if exe.startswith('a'):
    #             continue
    #         cleaned_exe=''.join([s for s in exe if not s.isdigit() and s!='.'])
    #         if i<len(exercises_1)-1 and exercises_1[i+1].startswith('a'):
    #             print(i,'是合适的题目')
    #             f.write(json.dumps({'id':i,'question_type':'multichoice','question':cleaned_exe,'choices':exercises_1[i+1]},ensure_ascii=False)+'\n')
    #         else:
    #             f.write(json.dumps({'id':i,'question_type':'freeform','question':cleaned_exe,'choices':None},ensure_ascii=False)+'\n')
    
    # exercises_1=extract_exercises_llm(cleaned_text)
    # print(exercises_1)
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_exercises_1.jsonl",'w',encoding='utf-8') as f:
    #     f.write(exercises_1)
        # for i,exe in enumerate(exercises_1):
        #     if exe.startswith('a'):
        #         continue
        #     cleaned_exe=''.join([s for s in exe if not s.isdigit() and s!='.'])
        #     if i<len(exercises_1)-1 and exercises_1[i+1].startswith('a'):
        #         print(i,'是合适的题目')
        #         f.write(json.dumps({'id':i,'question_type':'multichoice','question':cleaned_exe,'choices':exercises_1[i+1]},ensure_ascii=False)+'\n')
        #     else:
        #         f.write(json.dumps({'id':i,'question_type':'freeform','question':cleaned_exe,'choices':None},ensure_ascii=False)+'\n')
    questions=[]
    with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_exercises_1.jsonl",'r',encoding='utf-8') as f:
        for i,line in enumerate(f.readlines()):
            # print(i,line)
            questions.append(json.loads(line))
    cleaned_questions=[]
    idx=0
    for question in tqdm(questions):  
        question_type=question['question_type']
        question_text=question['question']
        res=clean_exercises_llm(question_text)  
        print(res)
        if 'True' in res:
            new_question={'id':idx,'question_type':question_type,'question':res[len('True,'):]}
            cleaned_questions.append(new_question)
            idx+=1
        elif 'False' in res:
            print(question_text)
            continue
        else:
            continue 
    
    with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_exercises_cleaned_llm.jsonl",'w',encoding='utf-8') as f:
        for cq in cleaned_questions:
            f.write(json.dumps(cq,ensure_ascii=False)+'\n')
        
    # exercises_2=extract_exercises(cleaned_text,"练习题")
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/statistic/statistic_exercises_2.jsonl",'w',encoding='utf-8') as f:
    #     for i,exe in tqdm(enumerate(exercises_2)):
    #         if exe.startswith('a'):
    #             continue
    #         cleaned_exe=''.join([s for s in exe if not s.isdigit() and s!='.'])
    #         is_legal,reason=check_legal_simple(question_type="freeform",question=cleaned_exe,choices=None)
    #         time.sleep(1)
    #         if is_legal:
    #             if i<len(exercises_1)-1 and exercises_1[i+1].startswith('a'):
    #                 f.write(json.dumps({'id':i,'question_type':'multichoice','question':cleaned_exe,'choices':exercises_1[i+1]},ensure_ascii=False)+'\n')
    #             else:
    #                 f.write(json.dumps({'id':i,'question_type':'freeform','question':cleaned_exe,'choices':None},ensure_ascii=False)+'\n')
    #         else:
    #             print(reason)
    
    
    
    