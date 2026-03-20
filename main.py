import json
import os
import time
import asyncio
from collections import defaultdict

from astrbot.api import logger
from astrbot.api.all import *
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api.message_components import Image, Plain, Forward, Node, Nodes
from astrbot.api.star import Context, Star, register


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cross_session_data")

SUMMARIZE_SYSTEM_PROMPT = """你是一个记忆助手。你的任务是将用户的对话内容提炼成简短的记忆摘要。
要求：
1. 用第三人称描述用户说了什么、表达了什么情绪或意图
2. 保留关键信息（情绪、事件、话题）
3. 每条摘要不超过50字
4. 只输出摘要文本，不要任何前缀或解释
5. 如果消息太短或没有实际意义（如"嗯"、"好的"），输出"无需记忆"
6. 如果有多条消息，合并为一条完整的摘要

示例：
输入：我今天加班到凌晨3点，好累啊，不想干了
输出：深夜加班到凌晨3点，感到疲惫，有离职念头

输入：刚买了个新手办，伊蕾娜的，超好看
输出：购入伊蕾娜手办，心情愉悦

输入：嗯
输出：无需记忆"""

GROUP_SUMMARIZE_PROMPT = """你是一个群聊记录员。将以下群聊对话总结为简短的场景描述。
要求：
1. 保留所有参与者的名字/昵称
2. 描述发生了什么事、谁说了什么关键内容
3. 控制在100字以内
4. 用第三人称客观描述
5. 如果对话没有实质内容（如纯表情、"嗯"、"好"），输出"无需记忆"

示例：
输入对话：
小明：我被欺负了，好烦
小红：怎么了？
小刚：别提了，他考试考的一团糟

输出：小明表示被欺负了心情烦躁，小刚透露原因是小明考试考砸了。"""

INJECT_TEMPLATE = """[跨会话记忆]
你记得这个用户（{sender_name}）最近在别处跟你聊过以下内容：
{memories}
你可以自然地在对话中引用这些记忆，比如调侃、关心或接着聊之前的话题。
不要生硬地说"你之前说过..."，要像真正记得一样自然提起。
如果当前话题跟这些记忆无关，就不用提。"""

GROUP_CONTEXT_INJECT_TEMPLATE = """[其他群聊动态]
你了解以下群最近发生的事情：
{group_contexts}
如果有人问起其他群的事情、提到某个群友、或话题相关，你可以自然引用这些信息来回答。
不要主动提起这些内容，只在被问到或明确相关时才引用。像是你真的参与了那些群聊一样自然回答。"""


class MemoryRecord:
    """单条跨会话记忆"""
    def __init__(self, sender_id: str, sender_name: str, session_id: str,
                 session_label: str, raw_message: str, summary: str,
                 timestamp: float, is_private: bool):
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.session_id = session_id
        self.session_label = session_label
        self.raw_message = raw_message
        self.summary = summary
        self.timestamp = timestamp
        self.is_private = is_private

    def to_dict(self):
        return vars(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class MemoryStore:
    """JSON 文件持久化存储 - 用户个人跨会话记忆"""
    def __init__(self, data_dir: str, max_records: int = 20, ttl_hours: int = 48):
        self.data_dir = data_dir
        self.max_records = max_records
        self.ttl_hours = ttl_hours
        self.store: dict[str, list[MemoryRecord]] = defaultdict(list)
        os.makedirs(data_dir, exist_ok=True)
        self._load()

    def _file_path(self):
        return os.path.join(self.data_dir, "memories.json")

    def _load(self):
        fp = self._file_path()
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for uid, records in data.items():
                    self.store[uid] = [MemoryRecord.from_dict(r) for r in records]
                total = sum(len(v) for v in self.store.values())
                logger.info(f"[cross_session] 加载了 {total} 条记忆，{len(self.store)} 个用户")
            except Exception as e:
                logger.error(f"[cross_session] 加载数据失败: {e}")

    def _save(self):
        try:
            data = {uid: [r.to_dict() for r in records]
                    for uid, records in self.store.items()}
            with open(self._file_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[cross_session] 保存数据失败: {e}")

    def add_record(self, record: MemoryRecord):
        uid = record.sender_id
        records = self.store[uid]
        records.append(record)
        cutoff = time.time() - self.ttl_hours * 3600
        records = [r for r in records if r.timestamp > cutoff]
        if len(records) > self.max_records:
            records = records[-self.max_records:]
        self.store[uid] = records
        self._save()

    def get_cross_memories(self, sender_id: str, current_session_id: str,
                           is_current_private: bool,
                           enable_p2g: bool, enable_g2p: bool, enable_g2g: bool,
                           max_records: int = 5) -> list[MemoryRecord]:
        records = self.store.get(sender_id, [])
        cutoff = time.time() - self.ttl_hours * 3600
        result = []
        for r in records:
            if r.timestamp < cutoff:
                continue
            if r.session_id == current_session_id:
                continue
            if is_current_private and r.is_private:
                continue
            if is_current_private and not r.is_private and not enable_g2p:
                continue
            if not is_current_private and r.is_private and not enable_p2g:
                continue
            if not is_current_private and not r.is_private and not enable_g2g:
                continue
            result.append(r)
        return result[-max_records:]

    def get_stats(self) -> dict:
        total = sum(len(v) for v in self.store.values())
        return {"users": len(self.store), "records": total}


class GroupChatLog:
    """群聊对话日志 - 记录各群的完整对话流（含所有群友）"""

    def __init__(self, data_dir: str, max_messages: int = 50,
                 max_summaries: int = 10, ttl_hours: int = 48):
        self.data_dir = data_dir
        self.max_messages = max_messages
        self.max_summaries = max_summaries
        self.ttl_hours = ttl_hours
        self.logs: dict[str, list[dict]] = defaultdict(list)
        self.summaries: dict[str, list[dict]] = defaultdict(list)
        os.makedirs(data_dir, exist_ok=True)
        self._load()

    def _file_path(self):
        return os.path.join(self.data_dir, "group_context.json")

    def _load(self):
        fp = self._file_path()
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logs = defaultdict(list, data.get("logs", {}))
                self.summaries = defaultdict(list, data.get("summaries", {}))
                total_msgs = sum(len(v) for v in self.logs.values())
                total_sums = sum(len(v) for v in self.summaries.values())
                logger.info(f"[cross_session] 群聊上下文: {len(self.logs)}个群, "
                            f"{total_msgs}条消息, {total_sums}条摘要")
            except Exception as e:
                logger.error(f"[cross_session] 加载群聊上下文失败: {e}")

    def _save(self):
        try:
            data = {"logs": dict(self.logs), "summaries": dict(self.summaries)}
            with open(self._file_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[cross_session] 保存群聊上下文失败: {e}")

    def add_message(self, session_id: str, session_label: str,
                    sender_name: str, sender_id: str, message: str):
        """添加一条群聊消息"""
        self.logs[session_id].append({
            "sender_name": sender_name,
            "sender_id": sender_id,
            "message": message[:300],
            "timestamp": time.time(),
            "session_label": session_label,
        })
        cutoff = time.time() - self.ttl_hours * 3600
        self.logs[session_id] = [
            m for m in self.logs[session_id] if m["timestamp"] > cutoff
        ][-self.max_messages:]
        self._save()

    def add_summary(self, session_id: str, session_label: str, summary: str):
        """添加一段群聊对话摘要"""
        self.summaries[session_id].append({
            "summary": summary,
            "session_label": session_label,
            "timestamp": time.time(),
        })
        cutoff = time.time() - self.ttl_hours * 3600
        self.summaries[session_id] = [
            s for s in self.summaries[session_id] if s["timestamp"] > cutoff
        ][-self.max_summaries:]
        self._save()

    def get_last_summary_ts(self, session_id: str) -> float:
        sums = self.summaries.get(session_id, [])
        return sums[-1]["timestamp"] if sums else 0

    def get_unsummarized_messages(self, session_id: str, last_summary_ts: float,
                                  min_count: int = 3) -> list[dict]:
        """获取某群自上次摘要后的新消息"""
        msgs = self.logs.get(session_id, [])
        new_msgs = [m for m in msgs if m["timestamp"] > last_summary_ts]
        return new_msgs if len(new_msgs) >= min_count else []

    def get_other_groups_for_inject(self, current_session_id: str,
                                    max_groups: int = 3,
                                    max_items: int = 5) -> list[dict]:
        """获取其他群的上下文。优先用摘要，没摘要则用原始消息"""
        cutoff = time.time() - self.ttl_hours * 3600
        all_sids = set(list(self.summaries.keys()) + list(self.logs.keys()))
        groups = []
        for sid in all_sids:
            if sid == current_session_id:
                continue
            valid_sums = [s for s in self.summaries.get(sid, [])
                          if s["timestamp"] > cutoff]
            valid_msgs = [m for m in self.logs.get(sid, [])
                          if m["timestamp"] > cutoff]
            if not valid_sums and not valid_msgs:
                continue
            label = ""
            if valid_sums:
                label = valid_sums[-1].get("session_label", sid)
            elif valid_msgs:
                label = valid_msgs[-1].get("session_label", sid)
            latest_ts = max(
                valid_sums[-1]["timestamp"] if valid_sums else 0,
                valid_msgs[-1]["timestamp"] if valid_msgs else 0,
            )
            groups.append({
                "session_id": sid,
                "session_label": label,
                "summaries": valid_sums[-max_items:] if valid_sums else [],
                "raw_messages": valid_msgs[-(max_items * 3):] if not valid_sums else [],
                "latest_ts": latest_ts,
            })
        groups.sort(key=lambda x: x["latest_ts"], reverse=True)
        return groups[:max_groups]

    def get_stats(self) -> dict:
        total_msgs = sum(len(v) for v in self.logs.values())
        total_sums = sum(len(v) for v in self.summaries.values())
        return {"groups": len(self.logs), "messages": total_msgs, "summaries": total_sums}


class MessageBuffer:
    """通用消息缓冲区，按 key 分组，静默一段时间后批量提交"""
    def __init__(self, debounce_seconds: float = 15.0):
        self.debounce_seconds = debounce_seconds
        self._buffers: dict = defaultdict(list)
        self._timers: dict = {}
        self._flush_callback = None

    def set_flush_callback(self, callback):
        self._flush_callback = callback

    async def add(self, key, item: dict):
        self._buffers[key].append(item)
        old_timer = self._timers.get(key)
        if old_timer and not old_timer.done():
            old_timer.cancel()
        self._timers[key] = asyncio.create_task(self._debounce_flush(key))

    async def _debounce_flush(self, key):
        try:
            await asyncio.sleep(self.debounce_seconds)
            await self._flush(key)
        except asyncio.CancelledError:
            pass

    async def _flush(self, key):
        items = self._buffers.pop(key, [])
        self._timers.pop(key, None)
        if items and self._flush_callback:
            await self._flush_callback(items)

    def get_pending_count(self) -> int:
        return sum(len(v) for v in self._buffers.values())

    def get_buffer_count(self) -> int:
        return len(self._buffers)


@register("cross_session_awareness", "Antigravity",
          "跨会话感知 - 跨群记忆 + 群聊上下文感知（v2.1）", "2.1.0")
class CrossSessionAwareness(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}

        max_records = self.config.get("max_records_per_user", 20)
        ttl_hours = self.config.get("record_ttl_hours", 48)
        self.inject_max = self.config.get("inject_max_records", 5)
        self.enable_p2g = self.config.get("enable_private_to_group", True)
        self.enable_g2p = self.config.get("enable_group_to_private", True)
        self.enable_g2g = self.config.get("enable_group_to_group", True)
        self.sid_blacklist = self.config.get("sid_blacklist", [])
        self.summarize_provider_id = self.config.get("summarize_provider_id", "")
        self.min_message_length = self.config.get("min_message_length", 4)
        debounce_sec = self.config.get("debounce_seconds", 15)

        # 群聊上下文配置
        self.enable_group_context = self.config.get("enable_group_context", True)
        group_max_msgs = self.config.get("group_context_max_messages", 50)
        self.group_context_max_groups = self.config.get("group_context_max_groups", 3)
        self.group_context_min_messages = self.config.get("group_context_min_messages", 3)
        group_debounce = self.config.get("group_context_debounce", 60)

        # 用户个人跨会话记忆
        self.store = MemoryStore(DATA_DIR, max_records, ttl_hours)

        # 群聊上下文日志
        self.group_log = GroupChatLog(DATA_DIR, max_messages=group_max_msgs,
                                      max_summaries=10, ttl_hours=ttl_hours)

        # 用户消息缓冲区（个人摘要用）
        self._buffer = MessageBuffer(debounce_seconds=debounce_sec)
        self._buffer.set_flush_callback(self._on_buffer_flush)

        # 群聊上下文缓冲区（群聊摘要用）
        self._group_buffer = MessageBuffer(debounce_seconds=group_debounce)
        self._group_buffer.set_flush_callback(self._on_group_buffer_flush)

        # 摘要队列和 worker
        self._summarize_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task = None

        # 合并转发暂存：session_id -> {"text": str, "sender_name": str, "ts": float}
        # 收到转发后不立即摘要，等群聊抖动结束时与后续讨论合并处理
        self._forward_pending: dict[str, dict] = {}
        # 转发暂存超时（秒），超时后单独摘要一次
        self._forward_pending_ttl: float = self.config.get("forward_pending_ttl", 300)

        user_stats = self.store.get_stats()
        group_stats = self.group_log.get_stats()
        logger.info(
            f"[cross_session] v2.0 初始化完成 | provider={self.summarize_provider_id} | "
            f"debounce={debounce_sec}s | group_ctx={'ON' if self.enable_group_context else 'OFF'} | "
            f"用户记忆: {user_stats} | 群聊上下文: {group_stats}"
        )

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        self._refresh_provider_options()

    def _refresh_provider_options(self):
        try:
            providers = self.context.get_all_providers()
            provider_ids = [""]
            for p in providers:
                pid = p.provider_config.get("id", "") if hasattr(p, "provider_config") else ""
                if pid:
                    provider_ids.append(pid)
            if len(provider_ids) <= 1:
                logger.debug("[cross_session] 无可用模型，跳过刷新")
                return
            schema_path = os.path.join(os.path.dirname(__file__), "_conf_schema.json")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            for key in ("summarize_provider_id", "vision_provider_id"):
                if key in schema and "options" in schema[key]:
                    schema[key]["options"] = provider_ids
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
            from astrbot.core.star.star import star_map
            for path, metadata in star_map.items():
                if "cross_session_awareness" in path and metadata.config and hasattr(metadata.config, "schema") and metadata.config.schema:
                    for key in ("summarize_provider_id", "vision_provider_id"):
                        if key in metadata.config.schema and "options" in metadata.config.schema[key]:
                            metadata.config.schema[key]["options"] = provider_ids
            logger.info(f"[cross_session] 已自动刷新模型选项: {len(provider_ids)-1} 个模型")
        except Exception as e:
            logger.warning(f"[cross_session] 刷新模型选项失败: {e}")

    async def _ensure_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._summarize_worker())

    async def _summarize_worker(self):
        while True:
            try:
                item = await asyncio.wait_for(self._summarize_queue.get(), timeout=300)
            except asyncio.TimeoutError:
                logger.debug("[cross_session] 摘要 worker 空闲超时退出")
                break
            except Exception:
                break

            item_type = item[0]
            if item_type == "user":
                _, sender_id, sender_name, session_id, session_label, message, ts, is_private, image_urls = item
                try:
                    summary = await self._call_llm_summarize(message, image_urls)
                    if summary and summary != "无需记忆":
                        record = MemoryRecord(
                            sender_id=sender_id, sender_name=sender_name,
                            session_id=session_id, session_label=session_label,
                            raw_message=message[:200], summary=summary,
                            timestamp=ts, is_private=is_private,
                        )
                        self.store.add_record(record)
                        logger.info(f"[cross_session] 📝 用户记忆: {sender_name}@{session_label} → {summary}")
                except Exception as e:
                    logger.error(f"[cross_session] 用户摘要失败: {e}")

            elif item_type == "group":
                _, session_id, session_label, conversation = item
                try:
                    summary = await self._call_group_summarize(conversation)
                    if summary and summary != "无需记忆":
                        self.group_log.add_summary(session_id, session_label, summary)
                        logger.info(f"[cross_session] 📋 群聊摘要: {session_label} → {summary}")
                    else:
                        logger.debug(f"[cross_session] 群聊对话无需记忆: {session_label}")
                except Exception as e:
                    logger.error(f"[cross_session] 群聊摘要失败: {e}")

    # ── 用户消息缓冲区 flush ──
    async def _on_buffer_flush(self, items: list[dict]):
        if not items:
            return
        last = items[-1]
        texts = [it["message"] for it in items if it["message"]]
        merged_message = "\n".join(texts) if texts else ""
        all_images = []
        for it in items:
            all_images.extend(it.get("image_urls", []))
        all_images = all_images[:3]
        if (not merged_message or len(merged_message.strip()) < self.min_message_length) and not all_images:
            return
        if len(items) > 1:
            logger.info(f"[cross_session] 🔄 合并 {last['sender_name']}@{last['session_label']} "
                        f"的 {len(items)} 条消息")
        await self._ensure_worker()
        await self._summarize_queue.put((
            "user",
            last["sender_id"], last["sender_name"],
            last["session_id"], last["session_label"],
            merged_message[:800], last["timestamp"],
            last["is_private"], all_images,
        ))

    # ── 群聊上下文缓冲区 flush ──
    async def _on_group_buffer_flush(self, items: list[dict]):
        """群聊静默后检查是否需要生成对话摘要"""
        if not items:
            return
        session_id = items[0]["session_id"]
        session_label = items[0]["session_label"]

        # 清理全局过期的 forward_pending（顺手扫描，避免泄漏）
        now = time.time()
        expired = [sid for sid, p in self._forward_pending.items()
                   if now - p["ts"] > self._forward_pending_ttl]
        for sid in expired:
            logger.debug(f"[cross_session] 转发暂存超时丢弃: {sid}")
            del self._forward_pending[sid]

        last_ts = self.group_log.get_last_summary_ts(session_id)
        new_msgs = self.group_log.get_unsummarized_messages(
            session_id, last_ts, min_count=self.group_context_min_messages)

        # ── 合并转发暂存处理 ──
        pending = self._forward_pending.get(session_id)
        forward_prefix = ""
        if pending:
            age = time.time() - pending["ts"]
            if age <= self._forward_pending_ttl:
                # pending 未超时：把转发内容作为前置上下文
                forward_prefix = (
                    f"【{pending['sender_name']} 分享的聊天记录】\n"
                    f"{pending['text']}\n\n【群友随后的讨论】\n"
                )
            # 无论如何清除 pending（超时或已用）
            del self._forward_pending[session_id]

        if not new_msgs and not forward_prefix:
            return

        # 格式化为对话文本
        conv_lines = [f"{m['sender_name']}：{m['message']}" for m in new_msgs]
        conversation = forward_prefix + "\n".join(conv_lines)

        logger.info(
            f"[cross_session] 🔄 准备生成群聊摘要: {session_label}, "
            f"{len(new_msgs)} 条新消息"
            + (f" + 附带转发上下文" if forward_prefix else "")
        )
        await self._ensure_worker()
        await self._summarize_queue.put(("group", session_id, session_label, conversation[:1500]))

    async def _call_llm_summarize(self, message: str, image_urls: list[str] = None) -> str:
        provider_id = self.summarize_provider_id
        if not provider_id:
            if image_urls:
                return message[:30] + " [含图片]" if message else "[发送了图片]"
            return message[:50] + ("..." if len(message) > 50 else "")

        vision_provider_id = self.config.get("vision_provider_id", "")
        use_provider = vision_provider_id if (image_urls and vision_provider_id) else provider_id
        try:
            prompt = message if message else "请描述这张图片的内容"
            if image_urls:
                prompt = f"用户发送了图片{'并说：' + message if message else ''}。请用简短的话总结用户的意图和图片内容。"
            resp = await self.context.llm_generate(
                chat_provider_id=use_provider, prompt=prompt,
                image_urls=image_urls or [], system_prompt=SUMMARIZE_SYSTEM_PROMPT,
            )
            if resp and resp.completion_text:
                return resp.completion_text.strip()[:100]
        except Exception as e:
            logger.warning(f"[cross_session] LLM 摘要调用失败 ({use_provider}): {e}")
            if image_urls:
                return message[:30] + " [含图片]" if message else "[发送了图片]"
            return message[:50] + ("..." if len(message) > 50 else "")
        return ""

    async def _call_group_summarize(self, conversation: str) -> str:
        """调用 LLM 生成群聊对话摘要"""
        provider_id = self.summarize_provider_id
        if not provider_id:
            # 无模型时直接截断作为摘要
            lines = conversation.split("\n")
            if len(lines) <= 3:
                return conversation[:150]
            return "\n".join(lines[:3])[:150] + "..."
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=conversation,
                system_prompt=GROUP_SUMMARIZE_PROMPT,
            )
            if resp and resp.completion_text:
                return resp.completion_text.strip()[:200]
        except Exception as e:
            logger.warning(f"[cross_session] 群聊摘要 LLM 调用失败: {e}")
            lines = conversation.split("\n")
            return "\n".join(lines[:3])[:150] + "..."
        return ""

    def _get_session_label(self, event: AstrMessageEvent) -> str:
        if event.is_private_chat():
            return "私聊"
        group_id = event.get_group_id()
        return f"群{group_id}" if group_id else "未知会话"

    def _is_blacklisted(self, event: AstrMessageEvent) -> bool:
        return event.unified_msg_origin in self.sid_blacklist

    @filter.on_llm_request()
    async def inject_memories(self, event: AstrMessageEvent, req: ProviderRequest):
        """在 LLM 请求前注入跨会话记忆 + 群聊上下文"""
        if self._is_blacklisted(event):
            return

        sender_id = event.get_sender_id()
        if not sender_id:
            return

        injected = 0

        # ── 1. 用户个人跨会话记忆（原有功能） ──
        memories = self.store.get_cross_memories(
            sender_id=sender_id,
            current_session_id=event.unified_msg_origin,
            is_current_private=event.is_private_chat(),
            enable_p2g=self.enable_p2g,
            enable_g2p=self.enable_g2p,
            enable_g2g=self.enable_g2g,
            max_records=self.inject_max,
        )
        if memories:
            sender_name = event.get_sender_name() or sender_id
            lines = []
            for m in memories:
                age_min = int((time.time() - m.timestamp) / 60)
                time_str = f"{age_min}分钟前" if age_min < 60 else f"{age_min // 60}小时前"
                loc = "私聊你时" if m.is_private else f"在{m.session_label}"
                lines.append(f"- {time_str}，{loc}：{m.summary}")
            context_block = INJECT_TEMPLATE.format(
                sender_name=sender_name, memories="\n".join(lines))
            req.system_prompt = (req.system_prompt or "") + "\n\n" + context_block
            injected += len(memories)

        # ── 2. 群聊上下文感知（新功能） ──
        if self.enable_group_context:
            group_contexts = self.group_log.get_other_groups_for_inject(
                current_session_id=event.unified_msg_origin,
                max_groups=self.group_context_max_groups,
            )
            if group_contexts:
                gc_lines = []
                for gc in group_contexts:
                    label = gc["session_label"]
                    if gc["summaries"]:
                        for s in gc["summaries"]:
                            age_min = int((time.time() - s["timestamp"]) / 60)
                            time_str = f"{age_min}分钟前" if age_min < 60 else f"{age_min // 60}小时前"
                            gc_lines.append(f"- [{label}] {time_str}: {s['summary']}")
                    elif gc["raw_messages"]:
                        # 无摘要时用原始消息
                        conv_parts = []
                        for m in gc["raw_messages"][-10:]:
                            conv_parts.append(f"  {m['sender_name']}: {m['message'][:80]}")
                        age_min = int((time.time() - gc["raw_messages"][-1]["timestamp"]) / 60)
                        time_str = f"{age_min}分钟前" if age_min < 60 else f"{age_min // 60}小时前"
                        gc_lines.append(f"- [{label}] {time_str}的对话:\n" + "\n".join(conv_parts))

                if gc_lines:
                    context_block = GROUP_CONTEXT_INJECT_TEMPLATE.format(
                        group_contexts="\n".join(gc_lines))
                    req.system_prompt = (req.system_prompt or "") + "\n\n" + context_block
                    injected += len(gc_lines)

        if injected > 0:
            logger.info(f"[cross_session] ✅ 已注入 {injected} 条上下文 "
                        f"(会话: {event.unified_msg_origin})")

    def _extract_nodes_text(self, nodes: list) -> str:
        """递归提取 Node 列表中的纯文本，返回 '昵称：内容' 格式"""
        lines = []
        for node in nodes:
            if not isinstance(node, Node):
                continue
            name = node.name or node.uin or "未知"
            parts = []
            for comp in (node.content or []):
                if isinstance(comp, Plain):
                    t = comp.text.strip()
                    if t:
                        parts.append(t)
                elif isinstance(comp, Image):
                    parts.append("[图片]")
                elif isinstance(comp, Node):
                    # 嵌套转发
                    parts.append(f"[转发消息: {self._extract_nodes_text([comp])}]")
                elif isinstance(comp, Nodes):
                    parts.append(f"[转发消息: {self._extract_nodes_text(comp.nodes)}]")
            if parts:
                lines.append(f"{name}：{''.join(parts)}")
        return "\n".join(lines)

    async def _fetch_forward_content(self, event: AstrMessageEvent, forward_id: str) -> str:
        """通过 OneBot API 拉取合并转发内容，返回对话文本（失败返回空串）"""
        try:
            bot = getattr(event, "bot", None)
            if bot is None:
                return ""
            result = await bot.call_action("get_forward_msg", message_id=forward_id)
            messages = result.get("messages") or result.get("data", {}).get("messages", [])
            lines = []
            for msg_data in messages:
                sender = msg_data.get("sender", {})
                name = sender.get("nickname") or sender.get("card") or str(sender.get("user_id", "未知"))
                content_list = msg_data.get("message") or []
                parts = []
                for seg in content_list:
                    seg_type = seg.get("type", "")
                    seg_data = seg.get("data", {})
                    if seg_type == "text":
                        t = seg_data.get("text", "").strip()
                        if t:
                            parts.append(t)
                    elif seg_type == "image":
                        parts.append("[图片]")
                    elif seg_type == "forward":
                        parts.append("[嵌套转发]")
                    elif seg_type in ("face", "mface"):
                        parts.append("[表情]")
                if parts:
                    lines.append(f"{name}：{''.join(parts)}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"[cross_session] 拉取合并转发内容失败 (id={forward_id}): {e}")
            return ""

    @filter.event_message_type(EventMessageType.ALL, priority=99990)
    async def capture_message(self, event: AstrMessageEvent):
        """捕获所有消息，存入缓冲区"""
        if self._is_blacklisted(event):
            return

        sender_id = event.get_sender_id()
        if not sender_id:
            return

        msg = event.get_message_str()
        sender_name = event.get_sender_name() or sender_id

        # 提取图片 URL
        image_urls = []
        # 检测合并转发
        forward_text = ""
        try:
            for comp in event.get_messages():
                if isinstance(comp, Image):
                    url = comp.url or comp.file
                    if url and (url.startswith("http://") or url.startswith("https://")):
                        image_urls.append(url)
                elif isinstance(comp, Forward):
                    # 只有 id，需调接口获取内容
                    if comp.id:
                        forward_text = await self._fetch_forward_content(event, comp.id)
                elif isinstance(comp, Nodes):
                    # 直接包含 nodes（少数平台/场景）
                    forward_text = self._extract_nodes_text(comp.nodes)
        except Exception:
            pass

        # ── 合并转发处理：暂存，等群聊抖动结束时与讨论合并 ──
        if forward_text:
            session_id = event.unified_msg_origin
            session_label = self._get_session_label(event)
            line_count = forward_text.count("\n") + 1
            logger.info(
                f"[cross_session] 📨 捕获合并转发: {sender_name}@{session_label}, "
                f"{line_count} 条消息，暂存等待后续讨论"
            )
            # 暂存转发内容，等群聊 debounce 结束时一起摘要
            self._forward_pending[session_id] = {
                "text": forward_text[:1000],
                "sender_name": sender_name,
                "ts": time.time(),
            }
            # 同时触发群聊 debounce（让抖动计时器重置，等待后续讨论）
            if self.enable_group_context:
                await self._group_buffer.add(session_id, {
                    "session_id": session_id,
                    "session_label": session_label,
                })
            return  # 合并转发单独处理，不走普通消息流程

        # 太短且无图片则跳过
        if (not msg or len(msg.strip()) < self.min_message_length) and not image_urls:
            return

        # 跳过命令
        if msg and (msg.startswith("/") or msg.startswith("!")):
            return

        session_id = event.unified_msg_origin
        session_label = self._get_session_label(event)

        # ── 1. 用户个人跨会话记忆缓冲区（原有） ──
        key = (sender_id, session_id)
        await self._buffer.add(key, {
            "sender_id": sender_id,
            "sender_name": sender_name,
            "session_id": session_id,
            "session_label": session_label,
            "message": (msg or "")[:500],
            "timestamp": time.time(),
            "is_private": event.is_private_chat(),
            "image_urls": image_urls[:3],
        })

        # ── 2. 群聊上下文记录（新功能，仅群聊） ──
        if not event.is_private_chat() and self.enable_group_context:
            self.group_log.add_message(
                session_id=session_id,
                session_label=session_label,
                sender_name=sender_name,
                sender_id=sender_id,
                message=(msg or "")[:300],
            )
            # 放入群聊缓冲区（触发 debounce 后生成摘要）
            await self._group_buffer.add(session_id, {
                "session_id": session_id,
                "session_label": session_label,
            })

    # ── 管理命令 ──
    @filter.command_group("跨群感知")
    def cross_session_cmd(self):
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @cross_session_cmd.command("状态")
    async def show_status(self, event: AstrMessageEvent):
        user_stats = self.store.get_stats()
        group_stats = self.group_log.get_stats()
        provider_id = self.summarize_provider_id or "未配置（使用原文截断）"
        debounce = self.config.get("debounce_seconds", 15)
        group_debounce = self.config.get("group_context_debounce", 60)
        yield event.plain_result(
            f"📊 跨会话感知 v2.0\n"
            f"━━━━ 用户记忆 ━━━━\n"
            f"记忆用户: {user_stats['users']}人\n"
            f"记忆总数: {user_stats['records']}条\n"
            f"摘要模型: {provider_id}\n"
            f"消息抖动: {debounce}秒\n"
            f"━━━━ 群聊上下文 ━━━━\n"
            f"启用: {'✅' if self.enable_group_context else '❌'}\n"
            f"监控群数: {group_stats['groups']}个\n"
            f"消息总数: {group_stats['messages']}条\n"
            f"摘要总数: {group_stats['summaries']}条\n"
            f"群聊抖动: {group_debounce}秒\n"
            f"━━━━ 方向控制 ━━━━\n"
            f"私聊→群聊: {'✅' if self.enable_p2g else '❌'}\n"
            f"群聊→私聊: {'✅' if self.enable_g2p else '❌'}\n"
            f"群聊→群聊: {'✅' if self.enable_g2g else '❌'}\n"
            f"━━━━ 缓冲区 ━━━━\n"
            f"用户缓冲: {self._buffer.get_buffer_count()}组/{self._buffer.get_pending_count()}条\n"
            f"群聊缓冲: {self._group_buffer.get_buffer_count()}组/{self._group_buffer.get_pending_count()}条\n"
            f"待处理队列: {self._summarize_queue.qsize()}条"
        )

    @cross_session_cmd.command("查看")
    async def show_memories(self, event: AstrMessageEvent):
        sender_id = event.get_sender_id()
        records = self.store.store.get(sender_id, [])
        if not records:
            yield event.plain_result("📭 暂无你的跨会话记忆")
            return
        lines = []
        for r in records[-10:]:
            age_min = int((time.time() - r.timestamp) / 60)
            time_str = f"{age_min}分钟前" if age_min < 60 else f"{age_min // 60}小时前"
            loc = "私聊" if r.is_private else r.session_label
            lines.append(f"• [{time_str}] {loc}: {r.summary}")
        yield event.plain_result(
            f"🧠 你的跨会话记忆（最近{len(lines)}条）\n" + "\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @cross_session_cmd.command("群聊动态")
    async def show_group_context(self, event: AstrMessageEvent):
        """查看所有群的摘要"""
        stats = self.group_log.get_stats()
        if stats["summaries"] == 0 and stats["messages"] == 0:
            yield event.plain_result("📭 暂无群聊上下文数据")
            return
        lines = [f"📋 群聊上下文 ({stats['groups']}个群, {stats['messages']}条消息, {stats['summaries']}条摘要)\n"]
        for sid, sums in self.group_log.summaries.items():
            if not sums:
                continue
            label = sums[-1].get("session_label", sid)
            lines.append(f"\n【{label}】")
            for s in sums[-3:]:
                age_min = int((time.time() - s["timestamp"]) / 60)
                time_str = f"{age_min}分钟前" if age_min < 60 else f"{age_min // 60}小时前"
                lines.append(f"  • {time_str}: {s['summary']}")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @cross_session_cmd.command("可用模型")
    async def list_providers(self, event: AstrMessageEvent):
        providers = self.context.get_all_providers()
        if not providers:
            yield event.plain_result("❌ 没有可用的 LLM 提供商")
            return
        lines = []
        for p in providers:
            pid = getattr(p, 'provider_id', '?')
            lines.append(f"• {pid}")
        yield event.plain_result(
            f"📋 可用 LLM 提供商（{len(lines)}个）\n"
            f"将 ID 填入插件设置的「摘要模型 ID」中\n"
            f"━━━━━━━━━━━━━━\n" + "\n".join(lines))
