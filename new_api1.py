import uvicorn
from fastapi import FastAPI, APIRouter
from typing import Dict, Any, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import os
import traceback
from openai import OpenAI
import httpx
import re

# 从环境变量获取AI配置
AI_API_KEY = os.environ.get("AI_MODEL_API_KEY", "xxxxxxxxxxxxxxxxxxx")
AI_API_URL = os.environ.get("AI_MODEL_API_URL", "https://api.deepseek.com/v1")
AI_MODEL_NAME = os.environ.get("AI_MODEL_NAME", "deepseek-chat")
AI_REQUEST_TIMEOUT = 180  # 请求超时时间（秒）

# 创建API路由器和FastAPI应用
router = APIRouter(prefix="/api", tags=["职业评测API"])
app = FastAPI(title="AI评测系统API", description="提供AI评测分析的API服务", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 修改结果存储为列表，以便按时间顺序存储
analysis_results = []


# 修改请求模型以匹配用户的JSON结构
class ExternalDataRequest(BaseModel):
    user_info: Dict[str, Any] = Field(..., description="用户信息")
    questions: List[Dict[str, Any]] = Field(..., description="问题列表")
    answers: Dict[str, Any] = Field(..., description="答案")


class APIResponse(BaseModel):
    code: int = Field(..., description="状态码")
    message: str = Field(..., description="状态消息")
    data: Optional[Dict[str, Any]] = Field(None, description="分析结果")


async def call_ai_model(prompt: str) -> Dict[str, Any]:
    """调用AI模型并解析结果"""
    try:
        print(f"\n===== 调用AI模型 =====")
        print(f"提示词长度: {len(prompt)} 字符")

        # 初始化OpenAI客户端
        client = OpenAI(api_key=AI_API_KEY, base_url=AI_API_URL)

        # 设置超时
        timeout = httpx.Timeout(AI_REQUEST_TIMEOUT, connect=180.0)

        # 调用API
        response = client.chat.completions.create(
            model=AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个返回标准JSON格式的API。请确保返回的内容是有效的JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=8190,
            timeout=AI_REQUEST_TIMEOUT
        )

        # 获取响应内容
        if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            print(f"API调用成功，响应长度: {len(content)} 字符")

            # 解析JSON
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content

            # 尝试修复和解析JSON
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)

            # 解析JSON
            data = json.loads(json_str)

            # 提取dimensions中的score值，创建dimension_scores字段
            if "dimensions" in data:
                dimension_scores = {}
                for dim_code, dim_detail in data["dimensions"].items():
                    if "score" in dim_detail:
                        dimension_scores[dim_code] = dim_detail["score"]
                data["dimension_scores"] = dimension_scores
                print(f"从dimensions中提取得分，创建dimension_scores: {dimension_scores}")

            return data
        else:
            raise Exception("API响应格式不正确")

    except Exception as e:
        print(f"调用AI模型出错: {str(e)}")
        print(traceback.format_exc())
        raise


@router.post("/analyze", response_model=APIResponse)
async def analyze_data(data: ExternalDataRequest):
    """处理外部数据并调用AI分析"""
    try:
        print("\n===== 接收到外部数据 =====")

        # 直接使用用户提供的user_info字段
        user_info = data.user_info

        # 转换问题和答案格式
        formatted_questions = []
        for i, q in enumerate(data.questions):
            question_id = q.get("id", str(i + 1))
            question_type = q.get("type", "single_choice")

            # 处理选项（如果是客观题）
            options = []
            if question_type in ["single_choice"] and "options" in q:
                for j, opt in enumerate(q.get("options", [])):
                    options.append({
                        "key": opt.get("key", str(j + 1)),
                        "content": opt.get("content", ""),
                        "score": opt.get("score", 0)
                    })

            formatted_questions.append({
                "id": question_id,
                "stem": q.get("stem", ""),
                "type": question_type,
                "options": options,
                "dimension": q.get("dimension", ""),
                "standard": q.get("standard", "")  # 添加standard字段
            })

        # 分离客观题和主观题
        objective_qa_pairs = []
        subjective_qa_pairs = []

        for q in formatted_questions:
            q_id = str(q.get("id", ""))
            if q_id in data.answers:
                question_type = q.get("type", "single_choice")

                # 创建问答对基本结构
                qa_pair = {
                    "question": q.get("stem", ""),
                    "dimension": q.get("dimension", ""),
                    "answer": data.answers[q_id],
                    "type": question_type,
                    "standard": q.get("standard", "")  # 添加standard字段
                }

                # 根据题目类型分类
                if question_type == "subject_question" or question_type == "career_transition" or question_type == "career_choice":
                    # 主观题
                    subjective_qa_pairs.append(qa_pair)
                else:
                    # 客观题，添加选项信息
                    qa_pair["options"] = q.get("options", [])
                    objective_qa_pairs.append(qa_pair)

        # 打印主观题和客观题的数量
        print(f"客观题数量: {len(objective_qa_pairs)}")
        print(f"主观题数量: {len(subjective_qa_pairs)}")

        # 添加用户实际选择的答案和分数到客观题数据中
        for qa in objective_qa_pairs:
            # 找到用户选择的选项
            user_answer = qa.get("answer", "")
            for option in qa.get("options", []):
                if option.get("key") == user_answer:
                    # 为每个问题添加用户选择和得分信息
                    qa["user_choice"] = {
                        "key": user_answer,
                        "content": option.get("content", ""),
                        "score": option.get("score", 0)
                    }
                    break

        # 计算每个维度的客观题得分
        objective_scores = {"R": 0, "E": 0, "A": 0, "D": 0, "Y": 0}
        for qa in objective_qa_pairs:
            dimension = qa.get("dimension", "")
            if dimension in objective_scores and "user_choice" in qa:
                objective_scores[dimension] += qa["user_choice"].get("score", 0)
        
        print("\n客观题各维度得分:")
        for dim, score in objective_scores.items():
            print(f"  {dim}: {score}")
        
        # 对主观题进行预处理分析
        subjective_analysis = {}
        subjective_scores = {"R": 0, "E": 0, "A": 0, "D": 0, "Y": 0}
        
        if subjective_qa_pairs:
            print("\n===== 开始分析主观题 =====")

            # 为每个维度收集相关的主观题回答
            dimension_responses = {"R": [], "E": [], "A": [], "D": [], "Y": []}

            # 遍历所有主观题
            for qa in subjective_qa_pairs:
                dim_value = qa.get("dimension", "")

                # 处理不同格式的维度值
                dims_to_process = []

                if isinstance(dim_value, list):
                    # 如果是列表，添加所有有效维度
                    for d in dim_value:
                        if d and d in dimension_responses:
                            dims_to_process.append(d)
                elif isinstance(dim_value, str) and dim_value:
                    # 如果是字符串且不为空，添加该维度
                    if dim_value in dimension_responses:
                        dims_to_process.append(dim_value)

                # 为每个识别到的维度添加问题
                for dim in dims_to_process:
                    dimension_responses[dim].append({
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "standard": qa.get("standard", "")
                    })

            # 只分析有数据的维度
            dimensions_with_data = [d for d in dimension_responses if dimension_responses[d]]
            print(f"将进行分析的维度: {dimensions_with_data}")

            # 检查是否每个维度恰好有一道题
            for dim in dimension_responses:
                question_count = len(dimension_responses[dim])
                if question_count != 1:
                    print(f"警告: 维度 {dim} 有 {question_count} 道题，期望值是1道题")

            for dim in dimensions_with_data:
                responses = dimension_responses[dim]
                if not responses:
                    continue
                # 构建主观题分析提示词
                subj_prompt = f"""
                你是一位专业的职业心理评测分析师。请分析以下用户对主观题的回答，并对用户回答的内容提取关键特质，评估其在{dim}维度上的表现，并为该维度的主观题答案打分。

                评分说明:
                - 本测评共有25道题，包括20道客观题和5道主观题
                - 主观题总共有5道，每个维度(R,E,A,D,Y)对应1道题
                - 每道主观题的满分是4分，最低分是0分
                - 这道题属于{dim}维度，你需要为它打0-4分（必须是整数）
                - 客观题总分80分，主观题总分20分，总计100分

                主观题回答：
                {json.dumps(responses, ensure_ascii=False, indent=2)}

                请使用以下评分标准：
                1. 逻辑性：答案的推理过程是否严密，结论是否合理。
                2. 结构性：答案的层次是否清晰，表述是否条理化。
                3. 完整性：是否涵盖标准答案的所有关键得分点。
                4. 准确性：答案内容是否与题目相关，是否符合要求。

                以下是加分项：
                1. 信息深度：回答的答案和内容高度相关，对问题，专业或行业有着深入的理解。
                2. 创新性：回答的答案有独特见解或新观点，展现创新思维。
                3. 数据支持：答案是否使用数据或者举例子来支持观点。
                4. 情感表达：用户的情感态度是否积极

                请使用以上标准，为该维度的这道题目的回答评分，评分范围为0-4分（必须整数）。

                以JSON格式返回，包括:该维度得分 (score) - 0-4之间的整数就好
               
                """

                try:
                    # 调用AI模型分析主观题
                    subj_result = await call_ai_model(subj_prompt)
                    subjective_analysis[dim] = subj_result
                    # 提取主观题得分
                    if "score" in subj_result:
                        subjective_scores[dim] = subj_result["score"]
                    print(f"维度 {dim} 主观题分析完成，评分: {subj_result.get('score', '未知')}")
                except Exception as e:
                    print(f"分析维度 {dim} 的主观题时出错: {str(e)}, 确保每个维度有且仅有一道主观题")
                    subjective_analysis[dim] = {
                        "analysis": f"分析失败: {str(e)}",
                        "score": 2,  # 默认中等分数
                        "key_traits": []
                    }
                    subjective_scores[dim] = 2  # 设置默认分数
        
        # 计算总分（客观题 + 主观题）
        total_scores = {}
        for dim in ["R", "E", "A", "D", "Y"]:
            total_scores[dim] = objective_scores[dim] + subjective_scores[dim]
        
        print("\n各维度总分（客观题 + 主观题）:")
        for dim, score in total_scores.items():
            print(f"  {dim}: {score}")

        # 准备提示信息，整合客观题和主观题分析
        prompt = f"""
        【重要】你是一位专业的职业心理评测分析师。你的任务是生成一份完整的1500字的心理测评报告。

        【数据说明】
        用户信息：
        {json.dumps(user_info, ensure_ascii=False, indent=2)}

        客观题答案：
        {json.dumps(objective_qa_pairs, ensure_ascii=False, indent=2)}

        主观题分析结果：
        {json.dumps(subjective_analysis, ensure_ascii=False, indent=2)}

        【维度得分】
        客观题得分：
        {json.dumps(objective_scores, ensure_ascii=False, indent=2)}
        
        主观题得分：
        {json.dumps(subjective_scores, ensure_ascii=False, indent=2)}
        
        总分（客观题+主观题）：
        {json.dumps(total_scores, ensure_ascii=False, indent=2)}

        【评分规则】
        1. 客观题总共20道，每题满分4分，总分80分
           - 客观题得分已经计算完成，在上面的客观题得分中提供

        2. 主观题总共5道，每题满分4分，总分20分
           - 每个维度(R,E,A,D,Y)对应1道主观题
           - 主观题得分已经计算完成，在上面的主观题得分中提供

        3. 最终得分已经计算完成，在上面的总分中提供，请直接使用这些分数

        【返回格式】
        必须返回以下JSON格式的dimensions对象，每个维度包含完整的评估信息：
        {{
             "R": {{
               "title": "决心/心理韧性",
               "score": 85,  
               "interpretation": "能力解读文本...",
               "abilityTitle": "心理韧性能力",
               "abilityDescription": "能力描述文本...",
               "improvementTitle": "提升建议",
               "suggestions": [
                 {{
                   "title": "建议类别1",
                   "items": ["具体建议1", "具体建议2"]
                 }},
                 {{
                   "title": "建议类别2",
                   "items": ["具体建议1", "具体建议2"]
                 }}
               ]
             }},
             "E": {{ "title": "就业准备度", "score": 70, "interpretation": "...", "abilityTitle": "...", "abilityDescription": "...", "improvementTitle": "提升建议", "suggestions": [] }},
             "A": {{ "title": "就业期望", "score": 65, "interpretation": "...", "abilityTitle": "...", "abilityDescription": "...", "improvementTitle": "提升建议", "suggestions": [] }},
             "D": {{ "title": "职业适应性", "score": 80, "interpretation": "...", "abilityTitle": "...", "abilityDescription": "...", "improvementTitle": "提升建议", "suggestions": [] }},
             "Y": {{ "title": "就业意愿", "score": 75, "interpretation": "...", "abilityTitle": "...", "abilityDescription": "...", "improvementTitle": "提升建议", "suggestions": [] }}
           }},
          dimension_scores: {{
            "R": 0,
            "E": 0,
            "A": 0,
            "D": 0,
            "Y": 0
          }}
        }}

        【字数要求】
        - interpretation字段: 200-300字
        - abilityDescription字段: 400-500字
        - improvementTitle字段: 200-300字
        - suggestions中每个item: 200-300字

        【其他要求】
        1. 所有文本内容必须使用中文
        2. 所有字段名必须使用英文
        3. 分数必须是整数
        4. 在评估维度时，请综合考虑客观题答案和主观题分析结果，将两个得分进行相加汇总
        5. 确保JSON格式正确，可以被直接解析
        """

        # 调用AI模型
        analysis_result = await call_ai_model(prompt)

        # 添加我们计算的分数到结果中
        analysis_result["objective_scores"] = objective_scores
        analysis_result["subjective_scores"] = subjective_scores
        analysis_result["dimension_scores"] = total_scores  # 使用我们计算的总分

        # 确保dimensions中的score与我们计算的总分一致
        if "dimensions" in analysis_result:
            for dim_code, dim_detail in analysis_result["dimensions"].items():
                if dim_code in total_scores:
                    dim_detail["score"] = total_scores[dim_code]

        # 添加主观题分析结果到返回数据中，方便调试
        analysis_result["subjective_analysis"] = subjective_analysis

        # 存储结果
        analysis_results.append({
            "username": user_info.get("username", "unknown"),
            "result": analysis_result,
            "timestamp": __import__('time').time()
        })

        # 打印解析结果
        print("\n===== 解析结果 =====")

        # 打印主观题分析结果
        print("\n主观题分析结果:")
        for dim, info in subjective_analysis.items():
            print(f"  维度 {dim}:")
            if isinstance(info, dict):
                for key, value in info.items():
                    if key == "analysis":
                        print(f"    {key}: {value[:50]}..." if value else "    分析: 无")
                    elif key == "key_traits" and isinstance(value, list):
                        print(f"    {key}: {', '.join(value[:3])}..." if value else "    特质: 无")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"    值类型: {type(info)}")

        # 打印维度得分
        if "dimension_scores" in analysis_result:
            print("\n维度得分:")
            for dim, score in analysis_result["dimension_scores"].items():
                print(f"  {dim}: {score}")

        return {
            "code": 200,
            "message": "分析成功",
            "data": analysis_result
        }

    except Exception as e:
        print(f"分析失败: {str(e)}")
        print(traceback.format_exc())
        return {
            "code": 500,
            "message": f"分析失败: {str(e)}",
            "data": None
        }


# 添加根路径说明
@app.get("/")
async def root():
    return {
        "message": "欢迎使用AI评测系统API",
        "documentation": "/docs"
    }


# 添加API路由
app.include_router(router)

if __name__ == "__main__":
    print("正在启动API服务，访问 http://localhost:8009/docs 查看API文档")
    uvicorn.run("new_api1:app", host="0.0.0.0", port=8009, reload=True)


