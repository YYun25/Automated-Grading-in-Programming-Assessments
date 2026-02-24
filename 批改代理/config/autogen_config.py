"""
進階 AutoGen 多代理人配置模組
實現更複雜的協作批改邏輯
"""

import os
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

logger = logging.getLogger(__name__)

class EnhancedMultiAgentGradingSystem:
    """增強版多代理人批改系統"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        self.grading_history = []
        self.current_task_id = None
        
    def _load_default_config(self) -> Dict[str, Any]:
        """載入預設配置"""
        return {
            "max_rounds": int(os.getenv("MAX_GRADING_ROUNDS", 8)),
            "timeout": int(os.getenv("GRADING_TIMEOUT", 300)),
            "temperature": {
                "grader": 0.1,
                "arbitrator": 0.05,
                "security": 0.0
            },
            "models": {
                "primary_grader": "gpt-4",
                "secondary_grader": "claude-3-sonnet-20240229", 
                "arbitrator": "gpt-4",
                "security_inspector": "gpt-3.5-turbo"
            }
        }
    
    def create_specialized_agents(self, grading_prompt: str, task_type: str = "general") -> Dict[str, Any]:
        """創建專門化的代理人"""
        
        # 基礎批改提詞
        base_grading_system = f"""
        {grading_prompt}
        
        ## 你的專業角色
        你是一位經驗豐富的程式設計教師，專精於C#程式語言教學與評分。
        
        ## 批改哲學
        - 以學習為導向，注重建設性回饋
        - 保持評分的客觀性與一致性
        - 考慮學生的學習階段與能力水平
        
        ## 回應格式要求
        請以結構化的方式回應，包含：
        1. 評分摘要
        2. 詳細分析
        3. 改進建議
        4. 得分明細表格
        """
        
        # GPT-4 主要批改專家
        gpt4_grader = AssistantAgent(
            name="GPT4_Expert_Grader",
            system_message=f"""
            {base_grading_system}
            
            ## 你的專長特色
            - 邏輯推理分析專家
            - 程式碼結構評估
            - 演算法效率判斷
            - 嚴謹的語法檢查
            
            ## 評分重點
            1. 邏輯正確性 (40%)
            2. 語法完整性 (30%)
            3. 程式碼品質 (20%)
            4. 創新解法 (10%)
            
            在協作過程中，請：
            - 提出具體的技術觀點
            - 指出邏輯錯誤與語法問題
            - 建議最佳實作方式
            - 與其他專家理性討論分歧
            """,
            llm_config={
                "config_list": [{
                    "model": self.config["models"]["primary_grader"],
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }],
                "temperature": self.config["temperature"]["grader"],
                "timeout": self.config["timeout"],
            }
        )
        
        # Claude 次要批改專家
        claude_grader = AssistantAgent(
            name="Claude_Expert_Grader",
            system_message=f"""
            {base_grading_system}
            
            ## 你的專長特色
            - 清晰的表達與解釋
            - 細緻的程式碼分析
            - 教學導向的回饋
            - 人性化的評語撰寫
            
            ## 評分重點
            1. 可讀性與風格 (35%)
            2. 邏輯清晰度 (35%)
            3. 錯誤處理 (20%)
            4. 文檔註解 (10%)
            
            在協作過程中，請：
            - 從教學角度提供見解
            - 注重學生的學習體驗
            - 提供具體的改進方向
            - 以同理心理解學生水平
            """,
            llm_config={
                "config_list": [{
                    "model": self.config["models"]["secondary_grader"],
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                }],
                "temperature": self.config["temperature"]["grader"],
                "timeout": self.config["timeout"],
            }
        )
        
        # 專業裁判代理
        arbitrator_agent = AssistantAgent(
            name="Senior_Arbitrator",
            system_message=f"""
            你是資深的教學評鑑專家與公正的裁判，負責協調不同專家的意見並做出最終決定。
            
            ## 裁判職責
            1. **分析專家意見** - 仔細評估各批改專家的論點
            2. **權衡評分標準** - 確保符合既定的評分準則
            3. **做出公正裁決** - 基於客觀證據給出最終評分
            4. **提供清晰說明** - 解釋裁決理由與評分依據
            5. **改善評分標準** - 發現標準不明確時提出改進建議
            
            ## 裁決原則
            - 以學生學習為中心
            - 保持評分的一致性
            - 基於具體證據而非主觀印象
            - 考慮評分標準的公平性
            
            ## 最終報告格式
            ```
            # 裁判最終決定
            
            ## 評分摘要
            - 總分：XX/100
            - 主要優點：...
            - 主要問題：...
            
            ## 專家意見分析
            - GPT-4專家觀點：...
            - Claude專家觀點：...
            - 分歧點分析：...
            
            ## 裁決理由
            [詳細說明評分依據與理由]
            
            ## 評分明細表格
            | 題號 | 配分 | 得分 | 評語 |
            |------|------|------|------|
            | ... | ... | ... | ... |
            
            ## 改進建議
            [具體的學習建議]
            ```
            """,
            llm_config={
                "config_list": [{
                    "model": self.config["models"]["arbitrator"],
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }],
                "temperature": self.config["temperature"]["arbitrator"],
                "timeout": self.config["timeout"],
            }
        )
        
        # 流程協調代理
        coordinator = UserProxyAgent(
            name="Grading_Coordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            system_message="""
            你是批改流程的協調者，負責：
            1. 引導批改專家進行充分討論
            2. 確保評分過程遵循既定標準
            3. 在需要時召喚裁判介入
            4. 記錄完整的協作過程
            """
        )
        
        return {
            "gpt4_grader": gpt4_grader,
            "claude_grader": claude_grader,
            "arbitrator": arbitrator_agent,
            "coordinator": coordinator
        }
    
    def create_adaptive_group_chat(self, agents: Dict[str, Any], task_complexity: str = "medium") -> GroupChat:
        """創建自適應群組聊天"""
        
        # 根據任務複雜度調整回合數
        complexity_rounds = {
            "simple": 4,
            "medium": 6, 
            "complex": 10
        }
        
        max_rounds = complexity_rounds.get(task_complexity, 6)
        
        # 自定義發言順序邏輯
        def custom_speaker_selection(last_speaker, groupchat):
            """智能發言者選擇邏輯"""
            messages = groupchat.messages
            
            if len(messages) == 0:
                return agents["coordinator"]
            
            last_message = messages[-1]
            last_speaker_name = last_message.get("name", "")
            
            # 協調者開始 -> 兩位專家輪流
            if last_speaker_name == "Grading_Coordinator":
                return agents["gpt4_grader"]
            
            # GPT-4 -> Claude
            elif last_speaker_name == "GPT4_Expert_Grader":
                return agents["claude_grader"]
            
            # Claude -> GPT-4 或裁判（如果需要）
            elif last_speaker_name == "Claude_Expert_Grader":
                # 檢查是否需要裁判介入
                if self._should_call_arbitrator(messages):
                    return agents["arbitrator"]
                else:
                    return agents["gpt4_grader"]
            
            # 裁判發言後結束
            elif last_speaker_name == "Senior_Arbitrator":
                return None
            
            return agents["gpt4_grader"]
        
        groupchat = GroupChat(
            agents=list(agents.values()),
            messages=[],
            max_round=max_rounds,
            speaker_selection_method=custom_speaker_selection,
            allow_repeat_speaker=False
        )
        
        return groupchat
    
    def _should_call_arbitrator(self, messages: List[Dict]) -> bool:
        """判斷是否需要裁判介入"""
        if len(messages) < 4:
            return False
        
        # 檢查最近的對話是否出現明顯分歧
        recent_messages = messages[-4:]
        
        # 簡單的分歧檢測邏輯
        disagreement_keywords = [
            "不同意", "disagree", "錯誤", "incorrect", 
            "應該是", "我認為", "但是", "however",
            "相反", "反對", "不對"
        ]
        
        disagreement_count = 0
        for msg in recent_messages:
            content = msg.get("content", "").lower()
            if any(keyword in content for keyword in disagreement_keywords):
                disagreement_count += 1
        
        # 如果分歧言論超過閾值，召喚裁判
        return disagreement_count >= 2
    
    def execute_collaborative_grading(self, 
                                    exam_content: str, 
                                    student_answer: str, 
                                    grading_prompt: str,
                                    task_id: str = None) -> Dict[str, Any]:
        """執行協作批改"""
        
        self.current_task_id = task_id or f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 分析任務複雜度
            task_complexity = self._analyze_task_complexity(exam_content, student_answer)
            
            # 創建專門化代理
            agents = self.create_specialized_agents(grading_prompt, task_complexity)
            
            # 創建自適應群組聊天
            groupchat = self.create_adaptive_group_chat(agents, task_complexity)
            
            # 創建群組管理器
            manager = GroupChatManager(
                groupchat=groupchat,
                llm_config={
                    "config_list": [{
                        "model": "gpt-4",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    }],
                    "temperature": 0.1,
                }
            )
            
            # 構建批改請求
            grading_request = self._build_grading_request(
                exam_content, student_answer, grading_prompt
            )
            
            # 開始協作批改
            logger.info(f"開始協作批改任務: {self.current_task_id}")
            
            chat_result = agents["coordinator"].initiate_chat(
                manager, 
                message=grading_request,
                max_turns=self.config["max_rounds"]
            )
            
            # 處理批改結果
            result = self._process_grading_result(groupchat.messages)
            
            # 記錄批改歷史
            self._record_grading_history(result)
            
            logger.info(f"協作批改完成: {self.current_task_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"協作批改過程出錯: {str(e)}")
            return {
                "final_score": 0,
                "reasoning": f"批改過程發生錯誤：{str(e)}",
                "error": True,
                "task_id": self.current_task_id
            }
    
    def _analyze_task_complexity(self, exam_content: str, student_answer: str) -> str:
        """分析任務複雜度"""
        
        # 簡單的複雜度評估邏輯
        complexity_indicators = {
            "code_lines": len(student_answer.split('\n')),
            "question_count": exam_content.count('Q') + exam_content.count('題'),
            "code_blocks": student_answer.count('{') + student_answer.count('}'),
        }
        
        total_score = (
            min(complexity_indicators["code_lines"] / 20, 3) +
            min(complexity_indicators["question_count"] / 3, 3) +
            min(complexity_indicators["code_blocks"] / 10, 3)
        )
        
        if total_score <= 3:
            return "simple"
        elif total_score <= 6:
            return "medium"
        else:
            return "complex"
    
    def _build_grading_request(self, exam_content: str, student_answer: str, grading_prompt: str) -> str:
        """構建批改請求"""
        
        return f"""
# 程式批改協作任務

## 任務說明
請兩位批改專家分別評估以下學生程式，並進行充分討論以達成共識。如果無法達成一致意見，將由裁判做出最終決定。

## 考試題目
```
{exam_content}
```

## 學生答案
```
{student_answer}
```

## 批改標準
{grading_prompt}

## 協作要求
1. **獨立評估** - 先各自給出初步評分與理由
2. **充分討論** - 針對分歧點進行深入討論
3. **尋求共識** - 努力達成一致的評分結果
4. **詳細記錄** - 記錄完整的批改過程與理由

請開始協作批改！
"""
    
    def _process_grading_result(self, chat_messages: List[Dict]) -> Dict[str, Any]:
        """處理批改結果"""
        
        # 提取最終評分
        final_score = self._extract_final_score(chat_messages)
        
        # 生成評分報告
        grading_report = self._generate_grading_report(chat_messages)
        
        # 分析協作品質
        collaboration_quality = self._analyze_collaboration_quality(chat_messages)
        
        return {
            "final_score": final_score,
            "grading_report": grading_report,
            "chat_history": chat_messages,
            "collaboration_quality": collaboration_quality,
            "task_id": self.current_task_id,
            "timestamp": datetime.now().isoformat(),
            "error": False
        }
    
    def _extract_final_score(self, messages: List[Dict]) -> float:
        """提取最終評分"""
        
        # 尋找最後的評分信息
        for message in reversed(messages):
            content = message.get("content", "")
            
            # 使用正則表達式提取分數
            import re
            
            # 尋找總分模式
            patterns = [
                r'總分[：:]\s*(\d+(?:\.\d+)?)',
                r'最終得分[：:]\s*(\d+(?:\.\d+)?)',
                r'得分[：:]\s*(\d+(?:\.\d+)?)',
                r'分數[：:]\s*(\d+(?:\.\d+)?)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
        
        # 如果找不到明確分數，返回0
        logger.warning("無法從對話中提取最終評分")
        return 0.0
    
    def _generate_grading_report(self, messages: List[Dict]) -> str:
        """生成評分報告"""
        
        report_sections = []
        
        # 添加標題
        report_sections.append("# 多代理人協作批改報告\n")
        
        # 添加任務信息
        report_sections.append(f"**任務ID:** {self.current_task_id}")
        report_sections.append(f"**批改時間:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append(f"**參與代理:** GPT-4專家, Claude專家, 資深裁判\n")
        
        # 添加批改過程
        report_sections.append("## 批改過程記錄\n")
        
        for i, message in enumerate(messages, 1):
            speaker = message.get("name", f"Agent_{i}")
            content = message.get("content", "")
            
            report_sections.append(f"### {speaker}")
            report_sections.append(f"```\n{content}\n```\n")
        
        return "\n".join(report_sections)
    
    def _analyze_collaboration_quality(self, messages: List[Dict]) -> Dict[str, Any]:
        """分析協作品質"""
        
        return {
            "total_rounds": len(messages),
            "agents_participated": len(set(msg.get("name", "") for msg in messages)),
            "average_message_length": sum(len(msg.get("content", "")) for msg in messages) / len(messages) if messages else 0,
            "arbitrator_involved": any("Arbitrator" in msg.get("name", "") for msg in messages),
            "consensus_reached": self._check_consensus_reached(messages)
        }
    
    def _check_consensus_reached(self, messages: List[Dict]) -> bool:
        """檢查是否達成共識"""
        
        # 簡化的共識檢測邏輯
        if not messages:
            return False
        
        # 如果裁判參與，視為達成共識
        if any("Arbitrator" in msg.get("name", "") for msg in messages):
            return True
        
        # 檢查最後幾條消息是否表達一致意見
        last_messages = messages[-3:] if len(messages) >= 3 else messages
        
        agreement_keywords = [
            "同意", "agree", "一致", "consensus", 
            "贊同", "支持", "認同", "確實"
        ]
        
        agreement_count = 0
        for msg in last_messages:
            content = msg.get("content", "").lower()
            if any(keyword in content for keyword in agreement_keywords):
                agreement_count += 1
        
        return agreement_count >= 2
    
    def _record_grading_history(self, result: Dict[str, Any]):
        """記錄批改歷史"""
        
        self.grading_history.append({
            "task_id": result["task_id"],
            "timestamp": result["timestamp"],
            "final_score": result["final_score"],
            "collaboration_quality": result["collaboration_quality"],
            "success": not result.get("error", False)
        })
        
        # 持久化到檔案（可選）
        try:
            import json
            history_file = "logs/grading_history.json"
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.warning(f"無法保存批改歷史: {str(e)}")
    
    def get_grading_statistics(self) -> Dict[str, Any]:
        """獲取批改統計信息"""
        
        if not self.grading_history:
            return {"message": "暫無批改記錄"}
        
        successful_gradings = [h for h in self.grading_history if h["success"]]
        
        return {
            "total_gradings": len(self.grading_history),
            "successful_gradings": len(successful_gradings),
            "success_rate": len(successful_gradings) / len(self.grading_history) * 100,
            "average_score": sum(h["final_score"] for h in successful_gradings) / len(successful_gradings) if successful_gradings else 0,
            "average_rounds": sum(h["collaboration_quality"]["total_rounds"] for h in successful_gradings) / len(successful_gradings) if successful_gradings else 0,
            "arbitrator_involvement_rate": sum(1 for h in successful_gradings if h["collaboration_quality"]["arbitrator_involved"]) / len(successful_gradings) * 100 if successful_gradings else 0
        }