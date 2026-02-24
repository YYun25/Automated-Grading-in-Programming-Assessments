#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量批改腳本 - 直接執行版本
不需要網頁界面，直接在程式裡指定檔案路徑並執行批改
結果和過程輸出到 "結果與過程.txt"
"""

import os
import sys
import uuid
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient, errors as mongo_errors

# 導入批改系統的核心模組
from app import (
    read_text, allowed_file,
    enhanced_split_by_question, split_by_question, extract_question_type,
    call_gpt_grader, call_claude_grader, call_gpt_consensus_diff, call_claude_consensus_diff, call_gemini_arbitration,
    call_gemini_similarity, call_gemini_similarity_consensus, normalize_items, _sort_items_by_id,
    i, calc_score_gap, decorate_comment_by_outcome, build_peer_diff_summary,
    render_final_table, build_fallback_feedback,
    run_prompt_autotune, create_or_bump_prompt,
    reset_claude_model_cache,
)

# 載入環境變數（優先載入 key.env，然後 .env）
load_dotenv("key.env")
load_dotenv()

# 檢查必要的環境變數
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not ANTHROPIC_API_KEY:
    print("⚠️  警告: ANTHROPIC_API_KEY 未設定，Claude 批改功能將無法使用")
    print("   請在 .env 檔案中設定 ANTHROPIC_API_KEY，或參考 .env.example")
    print()

if not OPENAI_API_KEY:
    print("⚠️  警告: OPENAI_API_KEY 未設定，GPT 批改功能將無法使用")
    print("   請在 .env 檔案中設定 OPENAI_API_KEY")
    print()

if not GEMINI_API_KEY:
    print("⚠️  警告: GEMINI_API_KEY 未設定，Gemini 仲裁功能將無法使用")
    print("   請在 .env 檔案中設定 GEMINI_API_KEY")
    print()

# MongoDB 連接（用於獲取評分提詞和記錄黑板訊息）
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "grading_blackboard")
mongo = MongoClient(MONGODB_URI)
db = mongo[MONGODB_DB]
col_prompts = db["grading_prompts"]
col_bbmsgs = db["blackboard_messages"]

# 建立索引
try:
    col_prompts.create_index([("subject", 1), ("version", -1)])
    col_bbmsgs.create_index([("task_id", 1), ("timestamp", -1)])
except Exception:
    pass

def get_latest_prompt(subject: str):
    """獲取最新版本的評分提詞"""
    return col_prompts.find_one({"subject": subject}, sort=[("version", -1)])

def log_prompt_blackboard(task_id: str, subject: str, action: str, content: str, payload=None):
    """記錄黑板訊息（立即寫入資料庫）"""
    col_bbmsgs.insert_one({
        "message_id": str(uuid.uuid4()),
        "task_id": task_id,
        "subject": subject,
        "type": action if action in ("initial_set","used","suggestion","updated","disagreement","consensus","arbitration_summary","quality_gate","similarity_check","question_flow","weakness_review","gpt_grade","claude_grade","consensus_round_enter","grading_complete","grading_start","detailed_results") else "info",
        "action": action,
        "content": content,
        "payload": payload,
        "created_by": "system" if action!="initial_set" else "user",
        "timestamp": datetime.now(timezone.utc)
    })

# ============================================================================
# 配置區域 - 請在這裡設定檔案路徑
# ============================================================================

# 科目設定（例如："C#", "Python", "Java" 等）
SUBJECT = "C#"

# ============================================================================
# 評分提詞設定 - 兩種方式擇一使用
# ============================================================================

# 方式 1: 從 MongoDB 讀取（預設）
# 請先在網頁系統 (http://localhost:5000) 中設定評分提詞
USE_MONGODB_PROMPT = True


# 題目資料夾路徑（自動讀取資料夾中的題目檔案）
QUESTION_FOLDER = "題目"  # 題目資料夾路徑

# 學生回答資料夾路徑（自動讀取資料夾中的所有答案檔案）
ANSWER_FOLDER = "學生回答"  # 學生回答資料夾路徑

# 結果與過程資料夾
RESULTS_FOLDER = "結果與過程"  # 批改過程結果資料夾


# ============================================================================
# 批改流程
# ============================================================================

class GradingLogger:
    """將批改過程記錄到檔案"""
    def __init__(self, output_file):
        self.output_file = output_file
        self.logs = []
        self.start_time = datetime.now()
        # 捕獲標準輸出，以便記錄所有 print 和 logger 輸出
        import sys
        import io
        self.original_stdout = sys.stdout
        self.captured_output = io.StringIO()
        
    def log(self, message, level="INFO"):
        """記錄訊息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        print(log_entry)  # 輸出到終端
        
    def log_section(self, title):
        """記錄區塊標題"""
        separator = "=" * 80
        self.logs.append("")
        self.logs.append(separator)
        self.logs.append(f"  {title}")
        self.logs.append(separator)
        
    def save(self):
        """儲存所有日誌到檔案"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # 添加總結
        self.logs.append("")
        self.logs.append("=" * 80)
        self.logs.append(f"批改完成 - 總耗時: {duration:.2f} 秒")
        self.logs.append("=" * 80)
        
        # 寫入檔案
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.logs))
        
        print(f"\n✅ 結果已儲存到: {self.output_file}")


def _trim_for_log(text: Optional[str], limit: int = 500) -> str:
    if not text:
        return ""
    text = str(text)
    return text if len(text) <= limit else (text[:limit] + "...")


def _log_section_with_label(logger: GradingLogger, label: str, content: Optional[str]):
    logger.log(label)
    trimmed = _trim_for_log(content)
    if trimmed:
        logger.log(trimmed)


SECTION_LABELS = ["核心判斷與詳細說明"]

def _split_analysis_sections(text: Optional[str]) -> dict[str, str]:
    core = ""
    if text:
        content = str(text)
        m = re.search(r"【核心判斷與詳細說明】(.*)", content, re.S)
        if m:
            extracted = m.group(1)
            extracted = re.split(r"\n\s*【", extracted, 1)[0].strip()
            core = extracted
        else:
            core = content.strip()
    return {"核心判斷與詳細說明": core}


def _log_formatted_outputs(logger: GradingLogger, *, comment, part1, part3):
    _log_section_with_label(logger, "【逐題評語】", comment)
    _log_section_with_label(logger, "【我的解答與驗證】", part1)
    sections = _split_analysis_sections(part3)
    core = sections.get("核心判斷與詳細說明", "").strip()
    if core:
        logger.log("【核心判斷與詳細說明】:")
        logger.log(core)

def grade_single_answer(question_file, answer_file, subject, logger, student_id=None):
    """批改單一答案檔案"""
    if student_id is None:
        student_id = os.path.basename(answer_file)
    
    logger.log_section(f"批改學生答案: {student_id}")
    logger.log(f"題目檔案: {question_file}")
    logger.log(f"答案檔案: {answer_file}")
    
    # 記錄批改開始到資料庫（先不記錄 task_id，因為還沒生成）
    # 這個會在生成 task_id 後記錄
    
    # 檢查檔案是否存在
    if not os.path.exists(question_file):
        logger.log(f"❌ 錯誤: 題目檔案不存在: {question_file}", "ERROR")
        return None
    
    if not os.path.exists(answer_file):
        logger.log(f"❌ 錯誤: 答案檔案不存在: {answer_file}", "ERROR")
        return None
    
    # 讀取檔案
    try:
        logger.log("正在讀取題目檔案...")
        exam_content = read_text(question_file)
        logger.log(f"題目內容長度: {len(exam_content)} 字元")
        
        logger.log("正在讀取答案檔案...")
        answer_content = read_text(answer_file)
        logger.log(f"答案內容長度: {len(answer_content)} 字元")
    except Exception as e:
        logger.log(f"❌ 讀取檔案失敗: {e}", "ERROR")
        return None
    
    # 獲取評分提詞
    if USE_MONGODB_PROMPT:
        logger.log(f"正在從 MongoDB 獲取科目「{subject}」的評分提詞...")
        prompt_doc = get_latest_prompt(subject)
        if not prompt_doc:
            logger.log(f"❌ 錯誤: 找不到科目「{subject}」的評分提詞", "ERROR")
            logger.log("   請選擇以下方式之一：", "ERROR")
            logger.log("   1. 在網頁系統 (http://localhost:5000) 中設定評分提詞", "ERROR")
            logger.log("   2. 將腳本中的 USE_MONGODB_PROMPT 設為 False，並設定 CUSTOM_PROMPT", "ERROR")
            return None
        logger.log(f"使用評分提詞版本: v{prompt_doc['version']}")
        prompt_content = prompt_doc["prompt_content"]
    else:
        logger.log("使用腳本中直接設定的評分提詞...")
        # 如果 USE_MONGODB_PROMPT = False，請在 CUSTOM_PROMPT 中設定提詞
        custom_prompt = globals().get('CUSTOM_PROMPT')
        if custom_prompt:
            prompt_content = custom_prompt
        else:
            logger.log("❌ 錯誤: 請設定 CUSTOM_PROMPT 變數", "ERROR")
            return None
        prompt_doc = {
            "subject": subject,
            "version": 1,
            "prompt_content": prompt_content
        }
    
    # 生成 task_id（格式：日期+時間+檔案名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 從答案檔案路徑提取檔案名（不含路徑和副檔名）
    answer_filename = Path(answer_file).stem  # 取得不含副檔名的檔案名
    # 清理檔案名中的特殊字元，只保留字母、數字、底線和連字號
    clean_filename = re.sub(r'[^\w\-]', '_', answer_filename)
    task_id = f"{timestamp}_{clean_filename}"
    logger.log(f"任務 ID: {task_id}")
    
    # 記錄批改開始到資料庫
    log_prompt_blackboard(
        task_id, subject, "grading_start",
        f"開始批改學生答案 - 學生ID: {student_id}",
        payload={
            "student_id": student_id,
            "question_file": question_file,
            "answer_file": answer_file,
            "timestamp": timestamp
        }
    )
    
    # 拆分題目和答案
    logger.log("正在拆分題目和答案...")
    exam_q_enhanced = enhanced_split_by_question(exam_content)
    ans_q = split_by_question(answer_content)
    
    # 獲取題號交集
    qids = sorted(
        set(exam_q_enhanced.keys()) & set(ans_q.keys()),
        key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 9999
    )
    
    if not qids:
        logger.log("⚠️  無法匹配題號，將整卷視為單一題目", "WARNING")
        qids = ["1"]
        exam_q_enhanced = {
            "1": {
                "content": exam_content,
                "max_score": 10.0,
                "question_type": extract_question_type(exam_content)
            }
        }
        ans_q = {"1": answer_content}
    
    logger.log(f"找到 {len(qids)} 個題目: {', '.join(qids)}")
    
    # 執行批改
    gpt_total = claude_total = final_total = 0
    gpt_items_all = []
    claude_items_all = []
    final_items_all = []
    
    consensus_round_qids = set()
    arbitration_qids = set()
    direct_consensus_qids = set()
    expected_scores = {}
    
    sim_threshold = 0.95
    SCORE_GAP_RATIO = 0.30
    CLAUDE_ITEMS_MAX_RETRY = max(1, int(os.getenv("CLAUDE_ITEMS_MAX_RETRY", "3")))
    CLAUDE_ITEMS_RETRY_DELAY = max(0.0, float(os.getenv("CLAUDE_ITEMS_RETRY_DELAY", "1.5")))
    
    def _claude_items_valid(res_payload):
        # 詳細調試：輸出完整的 res_payload 結構
        logger.log(f"🔍 Claude 驗證開始，res_payload keys: {list(res_payload.keys()) if isinstance(res_payload, dict) else 'not dict'}", "DEBUG")
        
        rubric = (res_payload or {}).get("rubric", {})
        logger.log(f"🔍 rubric 類型: {type(rubric)}, 內容: {rubric}", "DEBUG")
        
        items = rubric.get("items") if isinstance(rubric, dict) else None
        logger.log(f"🔍 items 類型: {type(items)}, 長度: {len(items) if isinstance(items, list) else 'N/A'}", "DEBUG")
        
        if not isinstance(items, list) or not items:
            logger.log(f"⚠️  Claude 驗證失敗：items 不是列表或為空，items={items}", "WARNING")
            return False
        
        first = items[0] or {}
        logger.log(f"🔍 第一個 item: {first}", "DEBUG")
        
        comment = str(first.get("comment", "")).strip()
        student_score = first.get("student_score")
        logger.log(f"🔍 comment 長度: {len(comment)}, student_score: {student_score} (類型: {type(student_score)})", "DEBUG")
        
        # 允許 comment 為空或 student_score 為 0，只要 student_score 不是 None
        is_valid = student_score is not None
        if not is_valid:
            logger.log(f"⚠️  Claude 驗證失敗：student_score 為 None，items={items[:1] if items else []}", "WARNING")
        else:
            logger.log(f"✅ Claude 驗證通過", "DEBUG")
        return is_valid
    
    def _ensure_claude_response_valid(initial_res, *, q_exam, q_ans, per_q_prompt, qid, peer_notes=None, phase_desc="初始批改"):
        res = initial_res
        logger.log(f"🔍 _ensure_claude_response_valid 開始（{phase_desc}｜題目 {qid}）", "DEBUG")
        logger.log(f"🔍 initial_res keys: {list(res.keys()) if isinstance(res, dict) else 'not dict'}", "DEBUG")
        
        if _claude_items_valid(res):
            logger.log(f"✅ Claude 驗證通過，直接返回（{phase_desc}｜題目 {qid}）", "DEBUG")
            return res
        
        logger.log(f"⚠️  Claude 第一次驗證失敗，嘗試從原始數據重新解析（{phase_desc}｜題目 {qid}）", "WARNING")
        
        # 驗證失敗時，嘗試從原始數據重新解析
        if res and res.get("raw"):
            try:
                from app import extract_json_best_effort, normalize_items
                import json
                import re
                raw_text = res.get("raw", "")
                logger.log(f"🔍 原始 raw 長度: {len(raw_text)}, 預覽: {raw_text[:200]}...", "DEBUG")
                
                # 改進解析：先嘗試標準方法
                raw_data = extract_json_best_effort(raw_text) or {}
                
                # 如果失敗，嘗試手動提取代碼塊中的 JSON（使用更強健的方法）
                if not raw_data or (isinstance(raw_data, dict) and not raw_data):
                    import re
                    # 嘗試提取 ```json ... ``` 中的內容
                    if "```json" in raw_text:
                        start = raw_text.find("```json") + 7
                        end = raw_text.find("```", start)
                        if end != -1:
                            json_text = raw_text[start:end].strip()
                            # 移除可能的語言標記（第一行如果不是 {）
                            lines = json_text.split('\n')
                            if len(lines) > 1 and not lines[0].strip().startswith('{'):
                                json_text = '\n'.join(lines[1:])
                            # 清理多餘的逗號
                            json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
                            try:
                                raw_data = json.loads(json_text)
                                logger.log(f"✅ 手動提取代碼塊 JSON 成功", "DEBUG")
                            except Exception as e:
                                logger.log(f"⚠️  手動提取失敗: {e}, JSON 預覽: {json_text[:200]}...", "WARNING")
                                # 如果還是失敗，嘗試找到第一個完整的 JSON 對象
                                try:
                                    first_brace = json_text.find('{')
                                    last_brace = json_text.rfind('}')
                                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                                        json_text_clean = json_text[first_brace:last_brace+1]
                                        json_text_clean = re.sub(r',\s*([}\]])', r'\1', json_text_clean)
                                        raw_data = json.loads(json_text_clean)
                                        logger.log(f"✅ 使用激進清理方法成功", "DEBUG")
                                except Exception as e2:
                                    logger.log(f"⚠️  激進清理也失敗: {e2}", "WARNING")
                    
                    # 如果還是失敗，嘗試提取 ``` ... ``` 中的內容
                    if not raw_data or (isinstance(raw_data, dict) and not raw_data):
                        if "```" in raw_text:
                            parts = raw_text.split("```")
                            for i in range(1, len(parts), 2):
                                if i < len(parts):
                                    json_text = parts[i].strip()
                                    # 移除可能的語言標記
                                    lines = json_text.split('\n')
                                    if len(lines) > 1 and not lines[0].strip().startswith('{'):
                                        json_text = '\n'.join(lines[1:])
                                    # 清理多餘的逗號
                                    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
                                    try:
                                        raw_data = json.loads(json_text)
                                        if raw_data:  # 確保不是空字典
                                            logger.log(f"✅ 手動提取普通代碼塊 JSON 成功", "DEBUG")
                                            break
                                    except Exception:
                                        # 嘗試激進清理
                                        try:
                                            first_brace = json_text.find('{')
                                            last_brace = json_text.rfind('}')
                                            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                                                json_text_clean = json_text[first_brace:last_brace+1]
                                                json_text_clean = re.sub(r',\s*([}\]])', r'\1', json_text_clean)
                                                raw_data = json.loads(json_text_clean)
                                                if raw_data:
                                                    logger.log(f"✅ 使用激進清理方法成功（普通代碼塊）", "DEBUG")
                                                    break
                                        except Exception:
                                            continue
                
                logger.log(f"🔍 解析後的 raw_data keys: {list(raw_data.keys()) if isinstance(raw_data, dict) else 'not dict'}", "DEBUG")
                
                rubric = raw_data.get("rubric", {})
                logger.log(f"🔍 解析後的 rubric 類型: {type(rubric)}, 內容: {rubric}", "DEBUG")
                
                if isinstance(rubric, dict):
                    items = normalize_items(rubric.get("items", []) or [])
                elif isinstance(rubric, list):
                    items = normalize_items(rubric)
                else:
                    items = []
                
                logger.log(f"🔍 normalize_items 後的 items 長度: {len(items)}, 內容: {items[:1] if items else []}", "DEBUG")
                
                if items and len(items) > 0:
                    first_item = items[0]
                    student_score = first_item.get("student_score")
                    logger.log(f"🔍 第一個 item 的 student_score: {student_score} (類型: {type(student_score)})", "DEBUG")
                    
                    if student_score is not None:
                        # 重新構建有效的結果
                        total = sum(i(x.get("student_score", 0)) for x in items if x.get("student_score") is not None)
                        res["rubric"] = {"items": items, "total_score": total}
                        res["score"] = total
                        logger.log(f"✅ Claude 從原始數據重新解析成功（{phase_desc}｜題目 {qid}），items 數量: {len(items)}, total: {total}", "INFO")
                        # 再次驗證
                        if _claude_items_valid(res):
                            return res
                        else:
                            logger.log(f"⚠️  重新解析後驗證仍然失敗", "WARNING")
                    else:
                        logger.log(f"⚠️  重新解析後 student_score 仍為 None", "WARNING")
                else:
                    logger.log(f"⚠️  重新解析後 items 為空", "WARNING")
            except Exception as e:
                logger.log(f"⚠️  Claude 從原始數據重新解析失敗：{e}", "WARNING")
                import traceback
                logger.log(f"   詳細錯誤: {traceback.format_exc()}", "WARNING")
        
        # 驗證失敗時，輸出詳細調試信息
        items = ((res or {}).get("rubric") or {}).get("items", [])
        logger.log(f"⚠️  Claude 驗證失敗詳情（{phase_desc}｜題目 {qid}）：", "WARNING")
        logger.log(f"   items 類型: {type(items)}, 長度: {len(items) if isinstance(items, list) else 'N/A'}", "WARNING")
        if items and len(items) > 0:
            first_item = items[0]
            logger.log(f"   第一個 item: {first_item}", "WARNING")
            logger.log(f"   comment: {repr(first_item.get('comment', ''))}", "WARNING")
            logger.log(f"   student_score: {repr(first_item.get('student_score'))}", "WARNING")
        last_res = res
        for retry_idx in range(1, CLAUDE_ITEMS_MAX_RETRY + 1):
            logger.log(
                f"⚠️  Claude 回傳 rubric.items 為空或格式異常（{phase_desc}｜題目 {qid}），啟動重試 {retry_idx}/{CLAUDE_ITEMS_MAX_RETRY}",
                "WARNING"
            )
            reset_claude_model_cache(demote_current=True)
            if CLAUDE_ITEMS_RETRY_DELAY > 0:
                time.sleep(CLAUDE_ITEMS_RETRY_DELAY)
            try:
                last_res = call_claude_grader(
                    q_exam, q_ans, per_q_prompt,
                    expected_item_ids=[qid],
                    peer_notes=peer_notes
                )
            except Exception as e:
                logger.log(f"❌ Claude 重試呼叫失敗：{e}", "ERROR")
                last_res = {
                    "agent": "claude",
                    "model": "retry_error",
                    "score": 0,
                    "feedback": f"Claude 重試失敗: {e}",
                    "rubric": {"items": []},
                    "raw": ""
                }
            if _claude_items_valid(last_res):
                logger.log(f"✅ Claude 重試成功（{phase_desc}｜題目 {qid}）")
                return last_res
        logger.log(
            f"❌ Claude 在 {phase_desc} 重試 {CLAUDE_ITEMS_MAX_RETRY} 次仍無有效評語（題目 {qid}），將以最後結果繼續流程",
            "ERROR"
        )
        last_res = last_res or {"agent": "claude", "score": 0, "rubric": {}, "feedback": "Claude 無法提供評語", "raw": ""}
        rubric = last_res.setdefault("rubric", {})
        items = rubric.setdefault("items", [])
        if not items:
            items.append({"item_id": qid, "max_score": 0, "student_score": 0, "comment": "Claude 無法提供評語"})
        elif not str(items[0].get("comment", "")).strip():
            items[0]["comment"] = "Claude 無法提供評語"
        return last_res
    
    for idx, qid in enumerate(qids, 1):
        logger.log("")
        logger.log(f"--- 批改題目 {idx}/{len(qids)}: {qid} ---")
        
        q_exam = exam_q_enhanced[qid]["content"]
        q_ans = ans_q[qid]
        expected_max_score = i(exam_q_enhanced[qid]["max_score"])
        # 不再從題目文本提取題型，改由批改代理人自行判斷
        type_hint = (
            "\n【題型判斷要求】\n"
            "請你先仔細閱讀題目內容，判斷此題屬於哪種題型（例如：問答題、程式實作題、選擇題、填空題等）。\n"
            "判斷完題型後，請優先套用批改標準中針對該題型的批改規則；\n"
            "若批改標準中沒有明確的題型分類，請選擇最接近的規則並在 comment 中說明你判斷的題型與依據。\n"
        )
        expected_scores[qid] = expected_max_score
        
        logger.log(f"題目配分: {expected_max_score} 分")
        
        per_q_prompt = (
            prompt_content +
            type_hint +
            f"\n\n【僅批改此題】請只針對『題目 {qid}』與其對應的學生答案評分，" +
            "不得參考其他題。rubric.items 僅需輸出此題一筆，item_id 請用題號。\n" +
            f"【重要】此題配分為 {expected_max_score} 分，請確保 max_score 設為 {expected_max_score}。\n"
            "【輸出格式限制】除了最外層要求的 JSON 結構外，你在【逐題評語】、【我的解答與驗證】、【核心判斷與詳細說明】等欄位中，"
            "一律不得輸出任何程式物件或結構化資料（例如 dict、JSON、鍵值對、表格物件等），而是要用一般的自然語言完整敘述。"
        )
        
        # 調用 GPT 和 Claude 批改
        try:
            logger.log("正在調用 GPT 批改...")
            logger.log("正在調用 Claude 批改...")
            with ThreadPoolExecutor(max_workers=2) as ex_pool:
                fut_g = ex_pool.submit(call_gpt_grader, q_exam, q_ans, per_q_prompt)
                fut_c = ex_pool.submit(call_claude_grader, q_exam, q_ans, per_q_prompt, expected_item_ids=[qid])
                try:
                    # 設置超時為 600 秒（10 分鐘），避免無限等待
                    gpt_res_q = fut_g.result(timeout=600)
                except KeyboardInterrupt:
                    logger.log("⚠️  收到中斷訊號，正在取消 GPT 批改任務...", "WARNING")
                    fut_g.cancel()
                    raise
                except Exception as e:
                    logger.log(f"❌ GPT 批改調用失敗: {e}", "ERROR")
                    fut_g.cancel()
                    raise
                
                try:
                    claude_res_raw = fut_c.result(timeout=600)
                except KeyboardInterrupt:
                    logger.log("⚠️  收到中斷訊號，正在取消 Claude 批改任務...", "WARNING")
                    fut_c.cancel()
                    raise
                except Exception as e:
                    logger.log(f"❌ Claude 批改調用失敗: {e}", "ERROR")
                    fut_c.cancel()
                    raise
                
                claude_res_q = _ensure_claude_response_valid(
                    claude_res_raw,
                    q_exam=q_exam,
                    q_ans=q_ans,
                    per_q_prompt=per_q_prompt,
                    qid=qid,
                    phase_desc="初始批改"
                )
            
            # 檢查 Claude 呼叫結果
            if not claude_res_q:
                logger.log("❌ Claude 返回結果為 None", "ERROR")
            elif claude_res_q.get("model") == "unavailable":
                logger.log(f"❌ Claude 模型不可用: {claude_res_q.get('feedback', '未知錯誤')}", "ERROR")
            elif claude_res_q.get("agent") == "claude":
                feedback = claude_res_q.get("feedback", "")
                if "不可用" in feedback or "無法使用" in feedback:
                    logger.log(f"❌ Claude 呼叫失敗: {feedback}", "ERROR")
                else:
                    logger.log(f"✅ Claude 呼叫成功，模型: {claude_res_q.get('model', '未知')}")
            
            # 強制單一項目並檢查分數，並避免回傳空的評語
            def _force_single_item_with_score_check(res, expected_score):
                items = normalize_items((res.get("rubric") or {}).get("items", [])[:1])
                if not items:
                    items = [{
                        "item_id": qid,
                        "max_score": expected_score,
                        "student_score": 0,
                        "comment": "模型未能產生有效評語，系統以 0 分並請人工覆核此題。"
                    }]
                
                items[0]["item_id"] = qid
                cur_max = i(items[0].get("max_score", 0))
                stu_raw = items[0].get("student_score", 0)
                
                def _parse_score(v):
                    if isinstance(v, (int, float)):
                        return float(v)
                    s = str(v).strip()
                    m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$', s)
                    if m:
                        num, den = float(m.group(1)), float(m.group(2))
                        return 0.0 if den == 0 else (num/den)
                    m2 = re.search(r'(\d+(?:\.\d+)?)', s)
                    return float(m2.group(1)) if m2 else 0.0
                
                stu = _parse_score(stu_raw)
                
                if cur_max <= 0:
                    if 0.0 <= stu <= 1.0:
                        stu = int(round(stu * expected_score))
                    else:
                        stu = int(round(max(0.0, min(stu, float(expected_score)))))
                elif cur_max != expected_score:
                    ratio = 0.0 if cur_max == 0 else (float(stu) / float(cur_max))
                    stu = int(round(ratio * expected_score))
                else:
                    stu = int(round(stu))
                
                items[0]["max_score"] = expected_score
                items[0]["student_score"] = stu

                # 確保 comment 不為空值或僅空白，避免後續流程出現「空評語」
                comment_str = str(items[0].get("comment", "") or "").strip()
                if not comment_str:
                    items[0]["comment"] = "模型回傳的評語為空或無法解析，系統自動補上此說明，請人工覆核此題。"
                
                res.setdefault("rubric", {}).update({"items": items, "total_score": stu})
                res["score"] = stu
                return res
            
            # 檢查 GPT 結果
            if not gpt_res_q:
                logger.log("❌ GPT 返回結果為 None", "ERROR")
                gpt_res_q = {"score": 0, "rubric": {"items": []}, "feedback": "GPT 呼叫失敗"}
            elif gpt_res_q.get("agent") == "gpt" and gpt_res_q.get("score") == 0 and not (gpt_res_q.get("rubric") or {}).get("items"):
                logger.log("⚠️  GPT 返回結果異常，可能呼叫失敗", "WARNING")
            
            gpt_res_q = _force_single_item_with_score_check(gpt_res_q, expected_max_score)
            claude_res_q = _force_single_item_with_score_check(claude_res_q, expected_max_score)
            
            g_score = i(gpt_res_q.get("score", 0))
            c_score = i(claude_res_q.get("score", 0))
            
            logger.log(f"GPT 得分: {g_score} / {expected_max_score}")
            logger.log(f"Claude 得分: {c_score} / {expected_max_score}")
            
            # 記錄 GPT 批改理由
            logger.log("")
            logger.log("--- GPT 批改理由 ---")
            gpt_comment = (gpt_res_q.get("rubric", {}).get("items", [{}])[0].get("comment", ""))
            gpt_part1 = gpt_res_q.get("part1_solution", "")
            gpt_part3 = gpt_res_q.get("part3_analysis", "")
            _log_formatted_outputs(
                logger,
                comment=gpt_comment,
                part1=gpt_part1,
                part3=gpt_part3,
            )
            
            # 記錄到 MongoDB（改回原本方式：content 簡短，詳細內容在 payload）
            log_prompt_blackboard(
                task_id, subject, "gpt_grade",
                f"GPT 批改題目 {qid}，得分：{g_score}/{expected_max_score}",
                payload={
                    "qid": qid,
                    "score": g_score,
                    "max_score": expected_max_score,
                    "comment": gpt_comment,
                    "part1_solution": gpt_part1,
                    "part3_analysis": gpt_part3
                }
            )
            
            # 記錄 Claude 批改理由
            logger.log("")
            logger.log("--- Claude 批改理由 ---")
            claude_comment = (claude_res_q.get("rubric", {}).get("items", [{}])[0].get("comment", ""))
            claude_part1 = claude_res_q.get("part1_solution", "")
            claude_part3 = claude_res_q.get("part3_analysis", "")
            _log_formatted_outputs(
                logger,
                comment=claude_comment,
                part1=claude_part1,
                part3=claude_part3,
            )
            
            # 記錄到 MongoDB（改回原本方式：content 簡短，詳細內容在 payload）
            log_prompt_blackboard(
                task_id, subject, "claude_grade",
                f"Claude 批改題目 {qid}，得分：{c_score}/{expected_max_score}",
                payload={
                    "qid": qid,
                    "score": c_score,
                    "max_score": expected_max_score,
                    "comment": claude_comment,
                    "part1_solution": claude_part1,
                    "part3_analysis": claude_part3
                }
            )
            
            # 如果 Claude 得分為 0，記錄詳細資訊
            if c_score == 0:
                claude_feedback = claude_res_q.get("feedback", "")
                claude_model = claude_res_q.get("model", "未知")
                claude_items = (claude_res_q.get("rubric") or {}).get("items", [])
                logger.log(f"⚠️  Claude 得分為 0，詳細資訊：", "WARNING")
                logger.log(f"   模型: {claude_model}", "WARNING")
                logger.log(f"   回饋: {claude_feedback}", "WARNING")
                logger.log(f"   項目數: {len(claude_items)}", "WARNING")
                if claude_res_q.get("raw"):
                    raw_preview = str(claude_res_q.get("raw", ""))[:300]
                    logger.log(f"   原始回應預覽: {raw_preview}...", "WARNING")
            
            # 語意相似度檢查
            sim = call_gemini_similarity(gpt_res_q, claude_res_q, threshold=sim_threshold)
            gap_abs, gap_ratio = calc_score_gap(g_score, c_score, expected_max_score)
            
            logger.log("")
            logger.log("--- 語意相似度檢查 ---")
            logger.log(f"相似度分數: {sim.get('score', 0):.2f} (門檻: {sim_threshold})")
            # 由程式判斷：score >= 0.95 視為相似
            is_similar_display = sim.get("score", 0) >= sim_threshold
            logger.log(f"是否相似: {'是' if is_similar_display else '否'}")
            sim_reason = sim.get("reason", "")
            if sim_reason:
                logger.log(f"評分理由: {sim_reason}")
            logger.log(f"分數差距: {gap_abs} / {expected_max_score} ({gap_ratio:.2%})")
            logger.log(f"差距門檻: {SCORE_GAP_RATIO:.0%}")
            
            # 記錄到 MongoDB（改回原本方式）
            log_prompt_blackboard(
                task_id, subject, "similarity_check",
                f"[題目 {qid}] 語意相似度：{sim.get('score', 0):.2f} ｜ 分數差：{gap_abs}/{expected_max_score}（{gap_ratio:.2%}）" + (f" ｜ 理由：{sim.get('reason', '')}" if sim.get('reason') else ""),
                payload={
                    "qid": qid,
                    "similarity_score": sim.get('score', 0),
                    "similar": sim.get('similar', False),
                    "reason": sim.get('reason', ''),
                    "gap_abs": gap_abs,
                    "gap_ratio": gap_ratio,
                    "threshold": sim_threshold,
                    "score_gap_threshold": SCORE_GAP_RATIO,
                    "gpt_score": g_score,
                    "claude_score": c_score
                }
            )
            
            outcome = None
            
            # 由程式判斷：score >= 0.95 視為相似
            is_similar = sim.get("score", 0) >= sim_threshold
            if is_similar and (gap_ratio < SCORE_GAP_RATIO):
                # 直接共識（保留小數點，最後再四捨五入）
                avg_score_raw = (g_score + c_score) / 2.0
                final_items_all.append({
                    "item_id": qid,
                    "max_score": expected_max_score,
                    "final_score": avg_score_raw,  # 保留原始小數
                    "final_score_rounded": i(avg_score_raw),  # 四捨五入後的分數
                    "comment": decorate_comment_by_outcome("語意一致且分數接近，採兩者平均。", "consensus")
                })
                final_total += avg_score_raw  # 累加原始小數
                logger.log("")
                logger.log(f"✅ 直接共識: {avg_score_raw:.1f} 分 (平均)")
                logger.log(f"   原因: 語意相似度 {sim.get('score', 0):.2f} ≥ {sim_threshold}，且分數差距 {gap_ratio:.2%} < {SCORE_GAP_RATIO:.0%}")
                outcome = "consensus"
                direct_consensus_qids.add(qid)
                
                # 記錄到 MongoDB（改回原本方式）
                log_prompt_blackboard(
                    task_id, subject, "consensus",
                    f"[題目 {qid}] 直接共識：平均 {avg_score_raw:.1f} 分（GPT: {g_score}, Claude: {c_score}）",
                    payload={
                        "qid": qid,
                        "avg_score": avg_score_raw,
                        "avg_score_rounded": i(avg_score_raw),
                        "gpt_score": g_score,
                        "claude_score": c_score,
                        "similarity_score": sim.get('score', 0),
                        "gap_ratio": gap_ratio,
                        "reason": f"語意相似度 {sim.get('score', 0):.2f} ≥ {sim_threshold}，且分數差距 {gap_ratio:.2%} < {SCORE_GAP_RATIO:.0%}"
                    }
                )
            else:
                # 進入共識回合（需要明確區分三種情況）
                # 由程式判斷：score >= 0.95 視為相似
                is_similar = sim.get("score", 0) >= sim_threshold
                is_gap_large = gap_ratio >= SCORE_GAP_RATIO
                
                if not is_similar and is_gap_large:
                    # 情況1：語意未達到標準，且分數差距太大
                    reason_enter = f"語意未達到標準（相似度 {sim.get('score', 0):.2f} < {sim_threshold}），且分數差距 {gap_ratio:.2%} ≥ {SCORE_GAP_RATIO:.0%}"
                elif is_similar and is_gap_large:
                    # 情況2：語意達到標準，但分數差距太大
                    reason_enter = f"語意達到標準（相似度 {sim.get('score', 0):.2f} ≥ {sim_threshold}），但分數差距 {gap_ratio:.2%} ≥ {SCORE_GAP_RATIO:.0%}"
                else:
                    # 情況3：語意未達到標準，但分數差異有達到門檻（分數差距小）
                    reason_enter = f"語意未達到標準（相似度 {sim.get('score', 0):.2f} < {sim_threshold}），但分數差距 {gap_ratio:.2%} < {SCORE_GAP_RATIO:.0%}（分數差異有達到門檻）"
                logger.log("")
                logger.log(f"--- 進入共識回合：{reason_enter} ---")
                consensus_round_qids.add(qid)
                
                # 記錄到 MongoDB（改回原本方式）
                log_prompt_blackboard(
                    task_id, subject, "consensus_round_enter",
                    f"[題目 {qid}] 進入共識回合，原因：{reason_enter}",
                    payload={
                        "qid": qid,
                        "reason": reason_enter,
                        "gpt_score": g_score,
                        "claude_score": c_score,
                        "gpt_comment": gpt_comment,
                        "claude_comment": claude_comment,
                        "similarity_score": sim.get('score', 0),
                        "gap_ratio": gap_ratio
                    }
                )
                
                agreed = False
                for round_idx in range(1):
                    logger.log("")
                    logger.log(f"--- 共識回合 {round_idx + 1}/1 ---")
                    
                    # 先用 Gemini 2.5 Pro 統整雙方差異，產生簡短摘要（給兩位代理參考）
                    # 傳入判斷資訊，讓 Gemini 根據不同情況（分數差異、語意差異、或兩者都有）進行分析
                    # ⚠️ 需求調整：暫停使用 Gemini 差異分析，避免增加額外的仲裁失敗風險
                    # peer_diff_summary = build_peer_diff_summary(
                    #     gpt_res_q, claude_res_q,
                    #     gpt_score=g_score, claude_score=c_score,
                    #     max_score=expected_max_score, gap_ratio=gap_ratio,
                    #     is_similar=is_similar, is_gap_large=is_gap_large
                    # )
                    
                    # 保存原批改結果（用於後續合併）
                    gpt_res_q_original = {
                        "score": g_score,
                        "rubric": {
                            "items": [{
                                "item_id": qid,
                                "max_score": expected_max_score,
                                "student_score": g_score,
                                "comment": (gpt_res_q.get("rubric", {}).get("items", [{}])[0].get("comment", ""))
                            }],
                            "total_score": g_score
                        },
                        "part1_solution": gpt_res_q.get("part1_solution", ""),
                        "part3_analysis": gpt_res_q.get("part3_analysis", "")
                    }
                    claude_res_q_original = {
                        "score": c_score,
                        "rubric": {
                            "items": [{
                                "item_id": qid,
                                "max_score": expected_max_score,
                                "student_score": c_score,
                                "comment": (claude_res_q.get("rubric", {}).get("items", [{}])[0].get("comment", ""))
                            }],
                            "total_score": c_score
                        },
                        "part1_solution": claude_res_q.get("part1_solution", ""),
                        "part3_analysis": claude_res_q.get("part3_analysis", "")
                    }
                    
                    # 提取 GPT 的完整內容（用於顯示給對方看）
                    g_cmt = (gpt_res_q.get("rubric", {}).get("items", [{}])[0].get("comment", ""))
                    g_part1 = gpt_res_q.get("part1_solution", "")
                    g_part3 = gpt_res_q.get("part3_analysis", "")
                    g_sections = _split_analysis_sections(g_part3)
                    g_core = g_sections.get("核心判斷與詳細說明", "")
                    
                    # 提取 Claude 的完整內容（用於顯示給對方看）
                    c_cmt = (claude_res_q.get("rubric", {}).get("items", [{}])[0].get("comment", ""))
                    c_part1 = claude_res_q.get("part1_solution", "")
                    c_part3 = claude_res_q.get("part3_analysis", "")
                    c_sections = _split_analysis_sections(c_part3)
                    c_core = c_sections.get("核心判斷與詳細說明", "")
                    
                    # 記錄 Gemini 的分析供使用者查看（不給批改代理人看）
                    # ⚠️ 已停用 Gemini 差異分析，保留原有紀錄格式於註解中
                    # logger.log("")
                    # logger.log("--- Gemini 2.5 Pro 分析未達成共識的原因（僅供使用者查看）---")
                    # logger.log(f"進入共識回合的原因：{reason_enter}")
                    # logger.log(f"Gemini 分析：{peer_diff_summary}")
                    #
                    # # 記錄到 MongoDB 供使用者查看
                    # log_prompt_blackboard(
                    #     task_id, subject, "consensus_round_analysis",
                    #     f"[題目 {qid}] Gemini 2.5 Pro 分析未達成共識的原因：{peer_diff_summary}",
                    #     payload={
                    #         "qid": qid,
                    #         "round": round_idx + 1,
                    #         "gemini_analysis": peer_diff_summary,
                    #         "reason_enter": reason_enter
                    #     }
                    # )
                    
                    # 為 GPT 構建 peer_notes（GPT 需要知道 Claude 的分數）
                    consensus_reason_note_gpt = f"""
【你的身份】你是 GPT 批改代理人。

【分數資訊】
- 你（GPT）給的分數：{g_score} 分
- 對方（Claude）給的分數：{c_score} 分

"""
                    
                    # 為 Claude 構建 peer_notes（Claude 需要知道 GPT 的分數）
                    consensus_reason_note_claude = f"""
【你的身份】你是 Claude 批改代理人。

【分數資訊】
- 你（Claude）給的分數：{c_score} 分
- 對方（GPT）給的分數：{g_score} 分

"""
                    
                    peer_notes_gpt = (
                        f"{consensus_reason_note_gpt}"
                        f"【共識回合 1/1 - 辯論式溝通】\n"
                        "現在進入「辯論式溝通」階段。這不是普通的閱讀對方內容，而是需要你以「辯論」的方式審視對方的批改觀點。"
                        "請仔細閱讀對方的【逐題評語】、【我的解答與驗證】與【核心判斷與詳細說明】。"
                        "請找出你與對方批改內容的差異，並針對差異進行辯論。\n\n"
                        f"【你的完整批改內容（GPT）】\n"
                        f"【逐題評語】{g_cmt}\n"
                        f"【我的解答與驗證】{g_part1}\n"
                        f"【核心判斷與詳細說明】{g_core}\n\n"
                        f"【對方的完整批改內容（Claude）】\n"
                        f"【逐題評語】{c_cmt}\n"
                        f"【我的解答與驗證】{c_part1}\n"
                        f"【核心判斷與詳細說明】{c_core}\n"
                    )
                    
                    peer_notes_claude = (
                        f"{consensus_reason_note_claude}"
                        f"【共識回合 1/1 - 辯論式溝通】\n"
                        "現在進入「辯論式溝通」階段。這不是普通的閱讀對方內容，而是需要你以「辯論」的方式審視對方的批改觀點。"
                        "請仔細閱讀對方的【逐題評語】、【我的解答與驗證】與【核心判斷與詳細說明】。"
                        "請找出你與對方批改內容的差異，並針對差異進行辯論。\n\n"
                        f"【對方的完整批改內容（GPT）】\n"
                        f"【逐題評語】{g_cmt}\n"
                        f"【我的解答與驗證】{g_part1}\n"
                        f"【核心判斷與詳細說明】{g_core}\n\n"
                        f"【你的完整批改內容（Claude）】\n"
                        f"【逐題評語】{c_cmt}\n"
                        f"【我的解答與驗證】{c_part1}\n"
                        f"【核心判斷與詳細說明】{c_core}\n"
                    )
                    
                    with ThreadPoolExecutor(max_workers=2) as ex_pool:
                        fut_g = ex_pool.submit(call_gpt_consensus_diff, q_exam, q_ans, per_q_prompt, peer_notes_gpt)
                        fut_c = ex_pool.submit(call_claude_consensus_diff, q_exam, q_ans, per_q_prompt, peer_notes_claude)
                        try:
                            # 設置超時為 600 秒（10 分鐘），避免無限等待
                            g_diff_res = fut_g.result(timeout=600)
                        except KeyboardInterrupt:
                            logger.log("⚠️  收到中斷訊號，正在取消 GPT 共識差異分析任務...", "WARNING")
                            fut_g.cancel()
                            raise
                        except Exception as e:
                            logger.log(f"❌ GPT 共識差異分析調用失敗: {e}", "ERROR")
                            fut_g.cancel()
                            raise
                        
                        try:
                            c_diff_res = fut_c.result(timeout=600)
                        except KeyboardInterrupt:
                            logger.log("⚠️  收到中斷訊號，正在取消 Claude 共識差異分析任務...", "WARNING")
                            fut_c.cancel()
                            raise
                        except Exception as e:
                            logger.log(f"❌ Claude 共識差異分析調用失敗: {e}", "ERROR")
                            fut_c.cancel()
                            raise
                    
                    # 處理 GPT 的差異分析結果
                    g_action = g_diff_res.get("action", "disagree")
                    if g_action == "agree":
                        # 如果同意：根據變更總結和更新後的關鍵內容，只更新原批改結果中被修改的部分
                        change_summary = g_diff_res.get("change_summary", "")
                        updated_content = g_diff_res.get("updated_key_content", {})
                        new_score = g_diff_res.get("new_score")
                        
                        # 更新分數（如果有調整）
                        if new_score is not None:
                            new_gpt_score = i(new_score)
                        else:
                            new_gpt_score = g_score
                        
                        # 更新 comment：原 comment + 變更總結 + 更新後的關鍵內容
                        original_comment = gpt_res_q_original["rubric"]["items"][0]["comment"]
                        updated_comment = updated_content.get("comment", "")
                        if updated_comment:
                            new_gpt_comment = f"{original_comment}\n\n【變更總結】\n{change_summary}\n\n【更新後的關鍵內容】\n{updated_comment}"
                        else:
                            new_gpt_comment = f"{original_comment}\n\n【變更總結】\n{change_summary}"
                        
                        # 更新 part1_solution 和 part3_analysis（只替換被修改的部分）
                        new_gpt_part1 = updated_content.get("part1_solution", "") or gpt_res_q_original["part1_solution"]
                        new_gpt_part3 = updated_content.get("part3_analysis", "") or gpt_res_q_original["part3_analysis"]
                    else:
                        # 如果不同意：完全保留原批改結果，只在 comment 中加入不同意的原因
                        disagree_reason = g_diff_res.get("disagree_reason", "")
                        original_comment = gpt_res_q_original["rubric"]["items"][0]["comment"]
                        new_gpt_comment = f"{original_comment}\n\n【不同意原因】\n{disagree_reason}"
                        new_gpt_score = g_score
                        new_gpt_part1 = gpt_res_q_original["part1_solution"]
                        new_gpt_part3 = gpt_res_q_original["part3_analysis"]
                    
                    # 處理 Claude 的差異分析結果
                    c_action = c_diff_res.get("action", "disagree")
                    if c_action == "agree":
                        # 如果同意：根據變更總結和更新後的關鍵內容，只更新原批改結果中被修改的部分
                        change_summary = c_diff_res.get("change_summary", "")
                        updated_content = c_diff_res.get("updated_key_content", {})
                        new_score = c_diff_res.get("new_score")
                        
                        # 更新分數（如果有調整）
                        if new_score is not None:
                            new_claude_score = i(new_score)
                        else:
                            new_claude_score = c_score
                        
                        # 更新 comment：原 comment + 變更總結 + 更新後的關鍵內容
                        original_comment = claude_res_q_original["rubric"]["items"][0]["comment"]
                        updated_comment = updated_content.get("comment", "")
                        if updated_comment:
                            new_claude_comment = f"{original_comment}\n\n【變更總結】\n{change_summary}\n\n【更新後的關鍵內容】\n{updated_comment}"
                        else:
                            new_claude_comment = f"{original_comment}\n\n【變更總結】\n{change_summary}"
                        
                        # 更新 part1_solution 和 part3_analysis（只替換被修改的部分）
                        new_claude_part1 = updated_content.get("part1_solution", "") or claude_res_q_original["part1_solution"]
                        new_claude_part3 = updated_content.get("part3_analysis", "") or claude_res_q_original["part3_analysis"]
                    else:
                        # 如果不同意：完全保留原批改結果，只在 comment 中加入不同意的原因
                        disagree_reason = c_diff_res.get("disagree_reason", "")
                        original_comment = claude_res_q_original["rubric"]["items"][0]["comment"]
                        new_claude_comment = f"{original_comment}\n\n【不同意原因】\n{disagree_reason}"
                        new_claude_score = c_score
                        new_claude_part1 = claude_res_q_original["part1_solution"]
                        new_claude_part3 = claude_res_q_original["part3_analysis"]
                    
                    # 構建最終結果
                    gpt_res_q = {
                        "score": new_gpt_score,
                        "rubric": {
                            "items": [{
                                "item_id": qid,
                                "max_score": expected_max_score,
                                "student_score": new_gpt_score,
                                "comment": new_gpt_comment
                            }],
                            "total_score": new_gpt_score
                        },
                        "part1_solution": new_gpt_part1,
                        "part3_analysis": new_gpt_part3
                    }
                    
                    claude_res_q = {
                        "score": new_claude_score,
                        "rubric": {
                            "items": [{
                                "item_id": qid,
                                "max_score": expected_max_score,
                                "student_score": new_claude_score,
                                "comment": new_claude_comment
                            }],
                            "total_score": new_claude_score
                        },
                        "part1_solution": new_claude_part1,
                        "part3_analysis": new_claude_part3
                    }
                    
                    g_score = new_gpt_score
                    c_score = new_claude_score
                    
                    # 記錄 GPT 批改理由（共識回合）
                    logger.log("")
                    logger.log(f"--- 共識回合 {round_idx + 1}/1 - GPT 代理 1 批改理由 ---")
                    gpt_comment_round = (gpt_res_q.get("rubric", {}).get("items", [{}])[0].get("comment", ""))
                    gpt_part1_round = gpt_res_q.get("part1_solution", "")
                    gpt_part3_round = gpt_res_q.get("part3_analysis", "")
                    if gpt_comment_round:
                        _log_formatted_outputs(
                            logger,
                            comment=gpt_comment_round,
                            part1=gpt_part1_round,
                            part3=gpt_part3_round,
                        )
                    
                    # 記錄到 MongoDB
                    log_prompt_blackboard(
                        task_id, subject, "gpt_grade",
                        f"GPT 批改題目 {qid}（共識回合 {round_idx + 1}/1），得分：{g_score}/{expected_max_score}",
                        payload={
                            "qid": qid,
                            "round": round_idx + 1,
                            "score": g_score,
                            "max_score": expected_max_score,
                            "comment": gpt_comment_round,
                            "part1_solution": gpt_part1_round,
                            "part3_analysis": gpt_part3_round
                        }
                    )
                    
                    # 記錄 Claude 批改理由（共識回合）
                    logger.log("")
                    logger.log(f"--- 共識回合 {round_idx + 1}/1 - claude 代理 2 批改理由 ---")
                    claude_comment_round = (claude_res_q.get("rubric", {}).get("items", [{}])[0].get("comment", ""))
                    claude_part1_round = claude_res_q.get("part1_solution", "")
                    claude_part3_round = claude_res_q.get("part3_analysis", "")
                    if claude_comment_round:
                        _log_formatted_outputs(
                            logger,
                            comment=claude_comment_round,
                            part1=claude_part1_round,
                            part3=claude_part3_round,
                        )
                    
                    # 記錄到 MongoDB
                    log_prompt_blackboard(
                        task_id, subject, "claude_grade",
                        f"Claude 批改題目 {qid}（共識回合 {round_idx + 1}/1），得分：{c_score}/{expected_max_score}",
                        payload={
                            "qid": qid,
                            "round": round_idx + 1,
                            "score": c_score,
                            "max_score": expected_max_score,
                            "comment": claude_comment_round,
                            "part1_solution": claude_part1_round,
                            "part3_analysis": claude_part3_round
                        }
                    )
                    
                    sim_after = call_gemini_similarity_consensus(
                        gpt_res_q_original, claude_res_q_original,
                        gpt_res_q, claude_res_q,
                        threshold=sim_threshold
                    )
                    gap_abs, gap_ratio = calc_score_gap(g_score, c_score, expected_max_score)
                    
                    logger.log("")
                    logger.log(f"本輪結果: GPT={g_score}, Claude={c_score}")
                    logger.log(f"相似度: {sim_after.get('score', 0):.2f}, 差距: {gap_ratio:.2%}")
                    sim_after_reason = sim_after.get("reason", "")
                    if sim_after_reason:
                        logger.log(f"評分理由: {sim_after_reason}")
                    
                    # 由程式判斷：score >= 0.95 視為相似
                    is_similar_after = sim_after.get("score", 0) >= sim_threshold
                    if is_similar_after and (gap_ratio < SCORE_GAP_RATIO):
                        avg_score_raw = (g_score + c_score) / 2.0
                        final_items_all.append({
                            "item_id": qid,
                            "max_score": expected_max_score,
                            "final_score": avg_score_raw,  # 保留原始小數
                            "final_score_rounded": i(avg_score_raw),  # 四捨五入後的分數
                            "comment": decorate_comment_by_outcome("共識回合達成語意一致且分數接近，採平均。", "consensus")
                        })
                        final_total += avg_score_raw  # 累加原始小數
                        logger.log(f"✅ 共識回合 {round_idx + 1} 達成: {avg_score_raw:.1f} 分")
                        outcome = "consensus"
                        agreed = True
                        
                        # 記錄到 MongoDB（改回原本方式）
                        log_prompt_blackboard(
                            task_id, subject, "consensus",
                            f"[題目 {qid}] 共識回合 {round_idx + 1}/1 達成：平均 {avg_score_raw:.1f} 分",
                            payload={
                                "qid": qid,
                                "round": round_idx + 1,
                                "avg_score": avg_score_raw,
                                "avg_score_rounded": i(avg_score_raw),
                                "gpt_score": g_score,
                                "claude_score": c_score,
                                "similarity_score": sim_after.get('score', 0),
                                "gap_ratio": gap_ratio
                            }
                        )
                        break
                    else:
                        logger.log(f"❌ 尚未達成共識 (相似度: {sim_after.get('score', 0):.2f}, 差距: {gap_ratio:.2%})")
                        
                        # 記錄到 MongoDB（改回原本方式）
                        log_prompt_blackboard(
                            task_id, subject, "disagreement",
                            f"[題目 {qid}] 共識回合 {round_idx + 1}/1 未達成共識",
                            payload={
                                "qid": qid,
                                "round": round_idx + 1,
                                "gpt_score": g_score,
                                "claude_score": c_score,
                                "similarity_score": sim_after.get('score', 0),
                                "gap_ratio": gap_ratio
                            }
                        )
                
                if not agreed:
                    # 仲裁
                    logger.log("")
                    logger.log("--- 交由 Gemini 仲裁 ---")
                    logger.log(f"GPT 最終得分: {g_score}")
                    logger.log(f"Claude 最終得分: {c_score}")
                    
                    # 檢查 Gemini 是否可用
                    if not GEMINI_API_KEY:
                        logger.log("⚠️  警告: GEMINI_API_KEY 未設定，無法使用 Gemini 仲裁", "WARNING")
                        logger.log("   請在 .env 或 key.env 檔案中設定 GEMINI_API_KEY", "WARNING")
                    else:
                        # 檢查 gemini_model 是否已初始化
                        try:
                            from app import gemini_model
                            if gemini_model is None:
                                logger.log("⚠️  警告: Gemini 模型未初始化，可能所有模型都無法使用", "WARNING")
                                logger.log("   請檢查 GEMINI_API_KEY 是否正確，或模型是否可用", "WARNING")
                        except Exception as e:
                            logger.log(f"⚠️  警告: 無法檢查 Gemini 模型狀態: {e}", "WARNING")
                    
                    try:
                        arb_q = call_gemini_arbitration(q_exam, q_ans, per_q_prompt, gpt_res_q, claude_res_q)
                        its = (arb_q.get("final_rubric") or {}).get("items", [])
                        if its:
                            it = its[0]
                            it["item_id"] = qid
                            it["comment"] = decorate_comment_by_outcome(it.get("comment", ""), "arbitration")
                            # 仲裁返回的分數可能是整數，轉為浮點數以保持一致性
                            arb_score = float(it.get("final_score", 0))
                            it["final_score"] = arb_score
                            final_items_all.append(it)
                            final_total += arb_score
                            
                            arb_reason = arb_q.get("reason", "")
                            arb_decision = arb_q.get("decision", "")
                            
                            logger.log(f"✅ 仲裁結果: {it.get('final_score', 0)} 分")
                            logger.log(f"仲裁決策: {arb_decision}")
                            logger.log(f"仲裁理由: {arb_reason}")
                            
                            # 如果仲裁結果顯示使用平均分，記錄詳細原因
                            if arb_decision == "average" or "不可用" in arb_reason or "使用平均" in arb_reason:
                                logger.log("⚠️  注意: 仲裁使用了平均分，可能原因：", "WARNING")
                                if not GEMINI_API_KEY:
                                    logger.log("   - GEMINI_API_KEY 未設定", "WARNING")
                                else:
                                    logger.log("   - Gemini 模型調用失敗（可能是 API 錯誤、配額用完或模型不可用）", "WARNING")
                                
                                # 如果有 Gemini 的回傳內容，顯示出來
                                gemini_raw = arb_q.get("gemini_raw_response")
                                if gemini_raw:
                                    logger.log("", "INFO")
                                    logger.log("📋 Gemini 回傳內容（即使調用失敗）：", "INFO")
                                    logger.log("─" * 80, "INFO")
                                    
                                    # 嘗試解析 JSON 以提供更詳細的分析
                                    try:
                                        import json
                                        raw_data = json.loads(gemini_raw)
                                        diagnostics = raw_data.get("diagnostics", {})
                                        
                                        # 分析 finish_reason
                                        finish_reason = diagnostics.get("finish_reason", "")
                                        finish_reason_code = diagnostics.get("finish_reason_code")
                                        if finish_reason:
                                            logger.log(f"🔍 完成原因: {finish_reason} (代碼: {finish_reason_code})", "INFO")
                                            if finish_reason_code == 2:
                                                logger.log("   ⚠️  注意: finish_reason=2 通常表示達到最大 token 限制", "WARNING")
                                            elif finish_reason_code == 3:
                                                logger.log("   ⚠️  注意: finish_reason=3 表示被安全過濾器阻擋", "WARNING")
                                        
                                        # 分析 safety_ratings
                                        safety_ratings = diagnostics.get("safety_ratings", [])
                                        if safety_ratings:
                                            logger.log("", "INFO")
                                            logger.log("🛡️  安全評級詳情:", "INFO")
                                            high_risk_count = 0
                                            for rating in safety_ratings:
                                                category = rating.get("category", rating.get("raw_category", "unknown"))
                                                probability = rating.get("probability", rating.get("raw_probability", "unknown"))
                                                prob_code = rating.get("probability_code")
                                                
                                                # 判斷是否為高風險
                                                is_high_risk = (
                                                    prob_code is not None and prob_code >= 2  # MEDIUM 或更高
                                                ) or (
                                                    "HIGH" in str(probability).upper() or 
                                                    "BLOCKED" in str(probability).upper()
                                                )
                                                
                                                if is_high_risk:
                                                    high_risk_count += 1
                                                    logger.log(f"   ⚠️  {category}: {probability} (高風險)", "WARNING")
                                                else:
                                                    logger.log(f"   ✓  {category}: {probability}", "INFO")
                                            
                                            if high_risk_count > 0:
                                                logger.log("", "INFO")
                                                logger.log(f"   ⚠️  檢測到 {high_risk_count} 個高風險安全評級，這可能是內容被阻擋的主要原因", "WARNING")
                                        
                                        logger.log("", "INFO")
                                        logger.log("完整診斷資訊:", "INFO")
                                    except:
                                        pass
                                    
                                    # 限制顯示長度，避免過長
                                    display_raw = gemini_raw[:2000] + ("..." if len(gemini_raw) > 2000 else "")
                                    logger.log(display_raw, "INFO")
                                    logger.log("─" * 80, "INFO")
                            
                            logger.log(f"最終評語: {it.get('comment', '')[:200]}..." if len(it.get('comment', '')) > 200 else f"最終評語: {it.get('comment', '')}")
                            
                            # 記錄到 MongoDB（改回原本方式）
                            log_prompt_blackboard(
                                task_id, subject, "arbitration_summary",
                                f"[題目 {qid}] 仲裁結果：{it.get('final_score', 0)} 分",
                                payload={
                                    "qid": qid,
                                    "final_score": it.get('final_score', 0),
                                    "decision": arb_decision,
                                    "reason": arb_reason,
                                    "gpt_score": g_score,
                                    "claude_score": c_score,
                                    "comment": it.get('comment', '')
                                }
                            )
                        else:
                            # 仲裁失敗，使用平均（保留小數點）
                            avg_score_raw = (g_score + c_score) / 2.0
                            final_items_all.append({
                                "item_id": qid,
                                "max_score": expected_max_score,
                                "final_score": avg_score_raw,  # 保留原始小數
                                "final_score_rounded": i(avg_score_raw),  # 四捨五入後的分數
                                "comment": decorate_comment_by_outcome("仲裁失敗，使用平均分。", "arbitration")
                            })
                            final_total += avg_score_raw  # 累加原始小數
                            logger.log(f"⚠️  仲裁失敗，使用平均: {avg_score_raw:.1f} 分")
                        outcome = "arbitration"
                        arbitration_qids.add(qid)
                    except Exception as e:
                        logger.log(f"❌ 仲裁錯誤: {e}", "ERROR")
                        avg_score_raw = (g_score + c_score) / 2.0
                        final_items_all.append({
                            "item_id": qid,
                            "max_score": expected_max_score,
                            "final_score": avg_score_raw,  # 保留原始小數
                            "final_score_rounded": i(avg_score_raw),  # 四捨五入後的分數
                            "comment": decorate_comment_by_outcome("仲裁失敗，使用平均分。", "arbitration")
                        })
                        final_total += avg_score_raw  # 累加原始小數
            
            # 記錄逐題結果
            gi = normalize_items((gpt_res_q.get("rubric") or {}).get("items"))[0] if (gpt_res_q.get("rubric") or {}).get("items") else {"item_id": qid, "max_score": expected_max_score, "student_score": 0, "comment": ""}
            ci = normalize_items((claude_res_q.get("rubric") or {}).get("items"))[0] if (claude_res_q.get("rubric") or {}).get("items") else {"item_id": qid, "max_score": expected_max_score, "student_score": 0, "comment": ""}
            
            if outcome in ("consensus", "arbitration"):
                gi["comment"] = decorate_comment_by_outcome(gi.get("comment", ""), outcome)
                ci["comment"] = decorate_comment_by_outcome(ci.get("comment", ""), outcome)
            
            gpt_items_all.append(gi)
            claude_items_all.append(ci)
            gpt_total += i(gi.get("student_score", 0))
            claude_total += i(ci.get("student_score", 0))
            
        except Exception as e:
            logger.log(f"❌ 批改題目 {qid} 時發生錯誤: {e}", "ERROR")
            import traceback
            logger.log(traceback.format_exc(), "ERROR")
            # 給預設分數
            gpt_items_all.append({
                "item_id": qid,
                "max_score": expected_max_score,
                "student_score": 0,
                "comment": f"批改錯誤：{str(e)}"
            })
            claude_items_all.append({
                "item_id": qid,
                "max_score": expected_max_score,
                "student_score": 0,
                "comment": f"批改錯誤：{str(e)}"
            })
            final_items_all.append({
                "item_id": qid,
                "max_score": expected_max_score,
                "final_score": 0,
                "comment": f"批改錯誤：{str(e)}"
            })
    
    # 生成最終結果
    logger.log("")
    logger.log_section("批改結果總結")
    
    # 計算最終總分（四捨五入）
    final_total_rounded = i(final_total)
    
    logger.log(f"任務 ID: {task_id}")
    logger.log(f"學生 ID: {student_id}")
    logger.log(f"GPT 總分: {gpt_total}")
    logger.log(f"Claude 總分: {claude_total}")
    if final_total != final_total_rounded:
        logger.log(f"最終總分: {final_total_rounded} 分 (原始分數: {final_total:.1f}，四捨五入)")
    else:
        logger.log(f"最終總分: {final_total_rounded} 分")
    logger.log(f"直接共識題數: {len(direct_consensus_qids)}")
    logger.log(f"共識回合題數: {len(consensus_round_qids)}")
    logger.log(f"仲裁題數: {len(arbitration_qids)}")
    
    # 記錄總結到 MongoDB（改回原本方式）
    log_prompt_blackboard(
        task_id, subject, "grading_complete",
        f"批改完成 - GPT: {gpt_total}, Claude: {claude_total}, 最終: {final_total_rounded} (原始: {final_total:.1f})",
        payload={
            "gpt_total": gpt_total,
            "claude_total": claude_total,
            "final_total": final_total_rounded,
            "final_total_raw": final_total,
            "direct_consensus_count": len(direct_consensus_qids),
            "consensus_round_count": len(consensus_round_qids),
            "arbitration_count": len(arbitration_qids)
        }
    )
    
    logger.log("")
    logger.log("--- 逐題詳細結果 ---")
    # 記錄逐題詳細結果到資料庫
    detailed_results = []
    for item in _sort_items_by_id(final_items_all):
        qid = item.get('item_id', 'N/A')
        final_score_raw = item.get('final_score', 0)
        max_score = item.get('max_score', 0)
        comment = item.get('comment', '無')
        
        # 每題顯示原始分數（保留小數點，不四捨五入）
        if isinstance(final_score_raw, float):
            logger.log(f"題目 {qid}: {final_score_raw:.1f} / {max_score} 分")
        else:
            logger.log(f"題目 {qid}: {final_score_raw} / {max_score} 分")
        if len(comment) > 300:
            logger.log(f"  評語: {comment[:300]}...")
        else:
            logger.log(f"  評語: {comment}")
        
        # 收集逐題結果（使用原始分數）
        detailed_results.append({
            "qid": qid,
            "final_score": final_score_raw,
            "max_score": max_score,
            "comment": comment
        })
    
    # 記錄逐題詳細結果到 MongoDB（改回原本方式）
    log_prompt_blackboard(
        task_id, subject, "detailed_results",
        f"逐題詳細結果 - 共 {len(detailed_results)} 題",
        payload={
            "student_id": student_id,
            "questions": detailed_results
        }
    )
    
    # === 自動提詞修改功能 ===
    logger.log("")
    logger.log_section("提詞自動優化檢查")
    
    try:
        entered_arbitration = len(arbitration_qids) > 0
        
        if not entered_arbitration:
            logger.log("本次所有題目皆未進入『仲裁』 → 跳過題詞自動優化。")
            logger.log(f"直接共識題數: {len(direct_consensus_qids)}")
            logger.log(f"共識回合題數: {len(consensus_round_qids)}")
            logger.log(f"仲裁題數: {len(arbitration_qids)}")
            log_prompt_blackboard(
                task_id, subject, "quality_gate",
                "本次所有題目皆未進入『仲裁』 → 跳過題詞自動優化。",
                payload={
                    "consensus_round_qids": sorted(list(consensus_round_qids)),
                    "arbitration_qids": [],
                    "direct_consensus_qids": sorted(list(direct_consensus_qids)),
                }
            )
        else:
            logger.log(f"檢測到進入仲裁的題目: {sorted(list(arbitration_qids))}")
            logger.log(f"直接共識題數: {len(direct_consensus_qids)}")
            logger.log(f"共識回合題數: {len(consensus_round_qids)}")
            logger.log("開始分析並優化提詞...")
            
            # 構建 GPT 和 Claude 的完整結果（用於提詞優化）
            gpt_res = {
                "agent": "gpt",
                "score": gpt_total,
                "rubric": {
                    "items": _sort_items_by_id(gpt_items_all),
                    "total_score": gpt_total
                },
            }
            
            claude_res = {
                "agent": "claude",
                "score": claude_total,
                "rubric": {
                    "items": _sort_items_by_id(claude_items_all),
                    "total_score": claude_total
                },
            }
            
            arbitration = {
                "final_score": final_total,
                "decision": "per_question",
                "reason": "每題各自共識/仲裁後彙整",
                "final_rubric": {
                    "items": _sort_items_by_id(final_items_all),
                    "total_score": final_total
                },
            }
            
            # 構建 context
            ctx = {
                "gpt": gpt_res,
                "claude": claude_res,
                "arbitration": arbitration,
                "expected_scores": expected_scores,
                "consensus_round_qids": sorted(list(consensus_round_qids)),
                "arbitration_qids": sorted(list(arbitration_qids)),
                "direct_consensus_qids": sorted(list(direct_consensus_qids)),
            }
            
            # 調用提詞自動優化
            auto = run_prompt_autotune(subject, prompt_content, ctx)
            
            if auto is not None:
                proposed = (auto.get("updated_prompt") or "").strip()
                reason = (auto.get("reason") or "").strip()
                diff_summary = (auto.get("diff_summary") or "").strip()
                
                # 記錄建議到資料庫
                log_prompt_blackboard(
                    task_id, subject, "suggestion",
                    f"Gemini 題詞建議：{diff_summary or '（無摘要）'}",
                    payload={
                        "proposed": proposed,
                        "reason": reason,
                        "diff_summary": diff_summary,
                        "consensus_round_qids": sorted(list(consensus_round_qids)),
                        "arbitration_qids": sorted(list(arbitration_qids)),
                    }
                )
                
                if proposed:
                    logger.log("")
                    logger.log("--- 提詞修改建議 ---")
                    logger.log(f"修改摘要: {diff_summary}")
                    logger.log(f"修改原因: {reason}")
                    logger.log("")
                    logger.log("正在自動套用新提詞...")
                    
                    # 直接更新提詞（不詢問）
                    pr2 = create_or_bump_prompt(subject, proposed, updated_by="gemini_autotune")
                    
                    logger.log(f"✅ 提詞已自動更新，版本升至 v{pr2['version']}")
                    logger.log(f"新提詞版本: v{pr2['version']}")
                    
                    # 記錄更新到資料庫
                    log_prompt_blackboard(
                        task_id, subject, "updated",
                        f"Gemini 已自動套用題詞，版本升至 v{pr2['version']}",
                        payload={
                            "source": "autotune_apply",
                            "diff_summary": diff_summary,
                            "new_version": pr2['version'],
                            "reason": reason
                        }
                    )
                else:
                    logger.log("Gemini 分析後認為無需修改提詞")
                    if reason:
                        logger.log(f"原因: {reason}")
            else:
                logger.log("⚠️  提詞自動優化失敗或返回空結果", "WARNING")
                
    except Exception as e:
        logger.log(f"❌ 提詞自動優化流程失敗: {e}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "ERROR")
    
    # 生成結果字典
    result = {
        "student_id": student_id,
        "task_id": task_id,
        "gpt_total": gpt_total,
        "claude_total": claude_total,
        "final_total": final_total,
        "gpt_items": gpt_items_all,
        "claude_items": claude_items_all,
        "final_items": final_items_all,
        "direct_consensus_count": len(direct_consensus_qids),
        "consensus_round_count": len(consensus_round_qids),
        "arbitration_count": len(arbitration_qids),
        "final_table_html": render_final_table(final_items_all, final_total)
    }
    
    return result

def scan_folder(folder_path):
    """掃描資料夾中的檔案（支援 .txt, .pdf, .docx）"""
    if not os.path.exists(folder_path):
        print(f"❌ 錯誤: 資料夾不存在: {folder_path}")
        return []
    
    if not os.path.isdir(folder_path):
        print(f"❌ 錯誤: 路徑不是資料夾: {folder_path}")
        return []
    
    files = []
    allowed_extensions = {'.txt', '.pdf', '.docx'}
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename.lower())
            if ext in allowed_extensions:
                files.append(file_path)
    
    # 按檔名排序
    files.sort()
    return files

def scan_answer_folder(folder_path):
    """掃描資料夾中的所有答案檔案（向後兼容）"""
    return scan_folder(folder_path)

def main():
    """主函數"""
    print("=" * 80)
    print("批量批改系統 - 直接執行版本")
    print("=" * 80)
    print()
    
    # 讀取題目檔案
    print(f"正在掃描題目資料夾: {QUESTION_FOLDER}")
    question_files = scan_folder(QUESTION_FOLDER)
    if not question_files:
        print(f"❌ 錯誤: 在題目資料夾中找不到任何題目檔案（支援 .txt, .pdf, .docx）")
        print(f"   資料夾路徑: {QUESTION_FOLDER}")
        return
    
    if len(question_files) > 1:
        print(f"⚠️  警告: 找到 {len(question_files)} 個題目檔案，將使用第一個: {os.path.basename(question_files[0])}")
    
    question_file = question_files[0]
    print(f"✅ 使用題目檔案: {os.path.basename(question_file)}")
    print()
    
    # 讀取學生回答檔案
    print(f"正在掃描學生回答資料夾: {ANSWER_FOLDER}")
    answer_files_list = scan_answer_folder(ANSWER_FOLDER)
    if not answer_files_list:
        print(f"❌ 錯誤: 在學生回答資料夾中找不到任何答案檔案（支援 .txt, .pdf, .docx）")
        print(f"   資料夾路徑: {ANSWER_FOLDER}")
        return
    
    print(f"✅ 找到 {len(answer_files_list)} 個答案檔案")
    for idx, f in enumerate(answer_files_list, 1):
        print(f"   {idx}. {os.path.basename(f)}")
    print()
    
    # 建立結果與過程資料夾
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    # 批改每個答案
    results = []
    for idx, answer_file in enumerate(answer_files_list, 1):
        print()
        print("=" * 80)
        print(f"處理答案檔案 {idx}/{len(answer_files_list)}: {os.path.basename(answer_file)}")
        print("=" * 80)
        
        # 使用檔案名（不含副檔名）作為學生ID
        student_id = os.path.splitext(os.path.basename(answer_file))[0]
        
        # 為每個學生建立獨立的日誌記錄器
        output_file = os.path.join(RESULTS_FOLDER, f"{student_id}批改過程.txt")
        logger = GradingLogger(output_file)
        
        logger.log_section("批量批改開始")
        logger.log(f"科目: {SUBJECT}")
        logger.log(f"題目檔案: {question_file}")
        logger.log(f"答案檔案: {answer_file}")
        logger.log(f"答案檔案數量: {len(answer_files_list)}")
        logger.log("")
        
        result = grade_single_answer(
            question_file,
            answer_file,
            SUBJECT,
            logger,
            student_id=student_id
        )
        
        if result:
            results.append(result)
            # 儲存該學生的批改過程
            logger.save()
            print(f"\n✅ {student_id} 的批改過程已儲存到: {output_file}")
        else:
            logger.log(f"❌ 跳過答案檔案: {answer_file}", "ERROR")
            logger.save()
            print(f"\n❌ {student_id} 的批改過程已儲存到: {output_file}（包含錯誤訊息）")
    
    # 總結
    print()
    print("=" * 80)
    print("所有批改任務完成")
    print("=" * 80)
    print(f"成功批改: {len(results)} / {len(answer_files_list)}")
    
    if results:
        print()
        print("--- 所有學生分數總結 ---")
        # 按照學生ID排序
        sorted_results = sorted(results, key=lambda x: str(x.get('student_id', '')))
        for result in sorted_results:
            try:
                student_id = result.get('student_id', '未知')
                if not student_id or student_id == '未知':
                    student_id = '未知學生'
                
                final_total_raw = result.get('final_total')
                
                # 確保 final_total_raw 是數字類型
                if final_total_raw is None:
                    # 嘗試從其他欄位計算
                    final_items = result.get('final_items', [])
                    if final_items:
                        final_total_raw = sum(float(item.get('final_score', 0)) for item in final_items)
                    else:
                        final_total_raw = 0
                elif not isinstance(final_total_raw, (int, float)):
                    try:
                        final_total_raw = float(final_total_raw)
                    except (ValueError, TypeError):
                        final_total_raw = 0
                else:
                    # 確保是浮點數
                    final_total_raw = float(final_total_raw)
                
                final_total_rounded = i(final_total_raw)
                if abs(final_total_raw - final_total_rounded) > 0.01:  # 有小數差異
                    print(f"{student_id}: {final_total_rounded} 分 (原始分數: {final_total_raw:.1f}，四捨五入)")
                else:
                    print(f"{student_id}: {final_total_rounded} 分")
            except Exception as e:
                student_id = result.get('student_id', '未知')
                import traceback
                print(f"{student_id}: 錯誤 - {str(e)}")
                print(f"  詳細錯誤: {traceback.format_exc()}")
    
    print()
    print("=" * 80)
    print("✅ 批改完成！所有結果已儲存到「結果與過程」資料夾")
    print("=" * 80)

if __name__ == "__main__":
    main()

