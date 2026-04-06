from __future__ import annotations

import asyncio
from collections import deque
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from astrbot.api import AstrBotConfig, logger, sp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.core.knowledge_base.kb_helper import KBHelper
from astrbot.core.provider.provider import EmbeddingProvider, RerankProvider
from astrbot.core.star import Context, Star
from astrbot.core.star.filter.command import GreedyStr

ALLOWED_HOSTS = {"thbwiki.cc", "www.thbwiki.cc"}
DEFAULT_BASE_URL = "https://thbwiki.cc/"
STOP_SECTION_TITLES = {
    "注释",
    "脚注",
    "参考资料",
    "参考文献",
    "外部链接",
    "参见",
    "导航菜单",
}
REMOVE_SELECTORS = (
    "script",
    "style",
    "noscript",
    ".mw-editsection",
    ".toc",
    ".catlinks",
    ".printfooter",
    ".reference",
    ".mw-references-wrap",
    ".thumb",
    ".gallery",
    ".navbox",
    ".vertical-navbox",
    ".metadata",
    ".noprint",
    ".mw-cite-backlink",
    ".mw-cite-backlink",
)
TOUHOU_KEYWORDS = (
    "东方",
    "touhou",
    "灵梦",
    "魔理沙",
    "咲夜",
    "妖梦",
    "幽幽子",
    "妹红",
    "天子",
    "阿求",
    "幻想乡",
    "博丽灵梦",
    "雾雨魔理沙",
    "琪露诺",
    "十六夜咲夜",
    "蕾米莉亚",
    "蕾咪",
    "芙兰朵露",
    "芙兰",
    "八云紫",
    "西行寺幽幽子",
    "魂魄妖梦",
    "射命丸文",
    "古明地觉",
    "古明地恋",
    "比那名居天子",
    "藤原妹红",
    "铃仙",
    "早苗",
    "阿求",
    "红魔馆",
    "永远亭",
    "守矢神社",
    "白玉楼",
    "香霖堂",
    "绯想天",
    "永夜抄",
    "红魔乡",
    "妖妖梦",
    "风神录",
    "地灵殿",
    "星莲船",
    "神灵庙",
    "辉针城",
    "绀珠传",
    "天空璋",
    "鬼形兽",
    "虹龙洞",
    "兽王园",
    "thbwiki",
)
DEFAULT_SYNC_EXCLUDE_TITLE_KEYWORDS = (
    "lostword",
    "大炮弹",
    "弹幕神乐",
    "play,doujin!",
    "comic market",
    "例大祭",
    "捏他列表",
    "branching paths",
    "啤酒",
    "黄昏酒场",
    "游戏为先还是酒为先",
    "二次创作以及使用规则",
)
DEFAULT_SYNC_EXCLUDE_CATEGORY_KEYWORDS = (
    "现实人物",
    "同人画师",
    "授权商业二次创作手机游戏",
)
TOUHOU_LLM_HINT = (
    "当前上下文已经启用东方Project知识库。"
    "当用户询问东方Project、角色、作品、设定、梗、地名、组织、时间线等相关问题时，"
    "应优先依据当前已注入的知识库内容回答。"
    "除非用户明确要求联网核验、抓网页、执行 Python 或写代码，否则不要优先改用网页搜索、浏览器抓取、"
    "Python 执行器或 Shell 工具来回答东方设定问题。"
)
PERSONA_BINDING_STORE_KEY = "touhou_kb_bound_persona_ids_v1"
MAX_SESSION_TOP_K = 20
MAX_CONTEXT_INJECTION_CHARS = 3000


@dataclass
class SyncTask:
    task_id: str
    entry: str
    limit: int
    kb_name: str
    status: str = "pending"
    current_url: str = ""
    current_title: str = ""
    discovered: int = 0
    imported: int = 0
    updated: int = 0
    failed: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    message: str = ""
    errors: list[str] = field(default_factory=list)

    def finish(self, status: str, message: str) -> None:
        self.status = status
        self.finished_at = time.time()
        self.message = message


@dataclass
class ParsedPage:
    url: str
    title: str
    categories: list[str]
    chunks: list[str]
    links: list[str]


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config or {}
        self.sync_tasks: dict[str, SyncTask] = {}
        self.session_last_task: dict[str, str] = {}
        self.background_jobs: dict[str, asyncio.Task] = {}
        self.global_last_task_id: str = ""
        self._kb_write_lock = asyncio.Lock()

    def _cfg_str(self, key: str, default: str) -> str:
        value = self.config.get(key, default)
        return str(value).strip() if value is not None else default

    def _cfg_int(self, key: str, default: int) -> int:
        value = self.config.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _cfg_float(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _kb_name(self) -> str:
        return self._cfg_str("kb_name", "东方Project知识库")

    def _kb_description(self) -> str:
        return self._cfg_str(
            "kb_description",
            "从 THBWiki 备用站导入的东方Project设定知识",
        )

    def _default_sync_entry(self) -> str:
        return self._cfg_str("default_entry_page", "东方Project")

    def _default_sync_limit(self) -> int:
        return max(1, min(self._cfg_int("default_sync_limit", 40), 500))

    def _normalize_persona_id(self, persona_id: str | None) -> str:
        normalized = str(persona_id or "").strip()
        if not normalized:
            raise ValueError("人格 ID 不能为空。")
        if normalized == "[%None]":
            raise ValueError("当前人格被显式设为空，无法绑定知识库。")
        if normalized not in {"default", "_chatui_default_"} and not self.context.persona_manager.get_persona_v3_by_id(normalized):
            raise ValueError(f"找不到人格：{normalized}")
        return normalized

    def _dedupe_text_list(
        self,
        values: list[str] | tuple[str, ...] | set[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _cfg_text_list(
        self,
        key: str,
        default: list[str] | tuple[str, ...],
    ) -> list[str]:
        value = self.config.get(key, default)
        if isinstance(value, (list, tuple, set)):
            parts = [str(item or "").strip() for item in value]
        else:
            parts = re.split(r"[\n,，;；]+", str(value or ""))
        return self._dedupe_text_list(parts)

    async def _get_bound_persona_ids(self) -> list[str]:
        raw = await self.get_kv_data(PERSONA_BINDING_STORE_KEY, [])
        if isinstance(raw, dict):
            raw = raw.get("persona_ids", [])
        if not isinstance(raw, list):
            return []
        return self._dedupe_text_list(raw)

    async def _set_bound_persona_ids(self, persona_ids: list[str]) -> list[str]:
        cleaned = self._dedupe_text_list(persona_ids)
        await self.put_kv_data(PERSONA_BINDING_STORE_KEY, cleaned)
        return cleaned

    async def _bind_persona_id(self, persona_id: str) -> list[str]:
        bindings = await self._get_bound_persona_ids()
        if persona_id not in bindings:
            bindings.append(persona_id)
        return await self._set_bound_persona_ids(bindings)

    async def _unbind_persona_id(self, persona_id: str) -> list[str]:
        bindings = await self._get_bound_persona_ids()
        bindings = [item for item in bindings if item != persona_id]
        return await self._set_bound_persona_ids(bindings)

    def _now_text(self, timestamp: float | None) -> str:
        if not timestamp:
            return "-"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

    def _normalize_intent_text(self, text: str | None) -> str:
        return "".join(str(text or "").strip().lower().split())

    def _is_touhou_intent(self, text: str | None) -> bool:
        normalized = self._normalize_intent_text(text)
        if not normalized:
            return False
        return any(keyword in normalized for keyword in TOUHOU_KEYWORDS)

    def _sanitize_kb_query(self, text: str | None) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"<Mnemosyne>.*?</Mnemosyne>", " ", cleaned, flags=re.S)
        cleaned = re.sub(r"<system_reminder>.*?</system_reminder>", " ", cleaned, flags=re.S)
        cleaned = re.sub(r"<Quoted Message>.*?</Quoted Message>", " ", cleaned, flags=re.S)
        cleaned = re.sub(r"<image_caption>.*?</image_caption>", " ", cleaned, flags=re.S)
        cleaned = re.sub(r"\[Image Attachment:[^\]]+\]", " ", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:400]

    def _normalize_top_k(self, top_k: int | None) -> int:
        fallback = max(1, min(self._cfg_int("session_top_k", 5), MAX_SESSION_TOP_K))
        if top_k is None:
            return fallback
        try:
            return max(1, min(int(top_k), MAX_SESSION_TOP_K))
        except (TypeError, ValueError):
            return fallback

    def _sync_exclude_title_keywords(self) -> list[str]:
        return self._cfg_text_list(
            "sync_exclude_title_keywords",
            DEFAULT_SYNC_EXCLUDE_TITLE_KEYWORDS,
        )

    def _sync_exclude_category_keywords(self) -> list[str]:
        return self._cfg_text_list(
            "sync_exclude_category_keywords",
            DEFAULT_SYNC_EXCLUDE_CATEGORY_KEYWORDS,
        )

    def _find_excluded_keyword(
        self,
        text: str,
        keywords: list[str],
    ) -> str:
        normalized = self._normalize_intent_text(text)
        for keyword in keywords:
            probe = self._normalize_intent_text(keyword)
            if probe and probe in normalized:
                return keyword
        return ""

    def _get_page_skip_reason(self, page: ParsedPage) -> str:
        title_keyword = self._find_excluded_keyword(
            page.title,
            self._sync_exclude_title_keywords(),
        )
        if title_keyword:
            return f"标题命中排除词：{title_keyword}"
        for category in page.categories:
            category_keyword = self._find_excluded_keyword(
                category,
                self._sync_exclude_category_keywords(),
            )
            if category_keyword:
                return f"分类命中排除词：{category_keyword}"
        return ""

    def _remove_competing_tools(self, request: ProviderRequest) -> None:
        if not getattr(request, "func_tool", None):
            return
        competing_tools = {
            "execute_python_code",
            "astrbot_execute_python",
            "astrbot_execute_shell",
            "fetch_url",
            "web_search",
            "search_web",
            "browser_exec",
            "browser_batch_exec",
        }
        for tool_name in competing_tools:
            try:
                request.func_tool.remove_tool(tool_name)
            except (AttributeError, KeyError, ValueError):
                continue

    async def _resolve_embedding_provider_id(self, preferred_id: str = "") -> str:
        provider_id = preferred_id.strip() or self._cfg_str(
            "default_embedding_provider_id",
            "",
        )
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider or not isinstance(provider, EmbeddingProvider):
                raise ValueError(
                    f"未找到可用的 Embedding Provider: {provider_id}。请先在 AstrBot 中配置 embedding 模型。"
                )
            return provider_id

        providers = self.context.get_all_embedding_providers()
        if not providers:
            raise ValueError(
                "当前没有可用的 Embedding Provider。请先在 AstrBot 里配置 embedding 模型，再执行初始化。"
            )
        return providers[0].meta().id

    def _get_all_rerank_providers(self) -> list[RerankProvider]:
        provider_manager = getattr(self.context, "provider_manager", None)
        if not provider_manager:
            return []
        providers = getattr(provider_manager, "rerank_provider_insts", [])
        if not isinstance(providers, list):
            return []
        return [
            provider for provider in providers if isinstance(provider, RerankProvider)
        ]

    async def _resolve_rerank_provider_id(
        self,
        preferred_id: str = "",
    ) -> str | None:
        provider_id = preferred_id.strip() or self._cfg_str(
            "default_rerank_provider_id",
            "",
        )
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider or not isinstance(provider, RerankProvider):
                raise ValueError(
                    f"未找到可用的 Rerank Provider: {provider_id}。请先在 AstrBot 中配置 rerank 模型，或清空插件里的默认 Rerank Provider ID。"
                )
            return provider_id

        providers = self._get_all_rerank_providers()
        if not providers:
            return None
        return providers[0].meta().id

    async def _ensure_kb(self, preferred_provider_id: str = "") -> tuple[KBHelper, bool]:
        kb_manager = self.context.kb_manager
        kb_name = self._kb_name()
        kb_helper = await kb_manager.get_kb_by_name(kb_name)
        if kb_helper:
            current_rerank_provider_id = getattr(
                kb_helper.kb,
                "rerank_provider_id",
                None,
            )
            if not current_rerank_provider_id:
                rerank_provider_id = await self._resolve_rerank_provider_id()
                if rerank_provider_id:
                    kb_helper = await kb_manager.update_kb(
                        kb_id=kb_helper.kb.kb_id,
                        kb_name=kb_helper.kb.kb_name,
                        description=kb_helper.kb.description,
                        emoji=kb_helper.kb.emoji,
                        embedding_provider_id=kb_helper.kb.embedding_provider_id,
                        rerank_provider_id=rerank_provider_id,
                        chunk_size=kb_helper.kb.chunk_size,
                        chunk_overlap=kb_helper.kb.chunk_overlap,
                        top_k_dense=kb_helper.kb.top_k_dense,
                        top_k_sparse=kb_helper.kb.top_k_sparse,
                        top_m_final=kb_helper.kb.top_m_final,
                    ) or kb_helper
                    logger.info(
                        "东方知识库已补充 Rerank Provider: kb=%s, rerank_provider_id=%s",
                        kb_helper.kb.kb_name,
                        rerank_provider_id,
                    )
            return kb_helper, False

        embedding_provider_id = await self._resolve_embedding_provider_id(
            preferred_provider_id
        )
        rerank_provider_id = await self._resolve_rerank_provider_id()
        kb_helper = await kb_manager.create_kb(
            kb_name=kb_name,
            description=self._kb_description(),
            emoji="☯️",
            embedding_provider_id=embedding_provider_id,
            rerank_provider_id=rerank_provider_id,
            chunk_size=512,
            chunk_overlap=50,
            top_k_dense=50,
            top_k_sparse=50,
            top_m_final=self._normalize_top_k(None),
        )
        return kb_helper, True

    async def _get_touhou_kb(self) -> KBHelper | None:
        return await self.context.kb_manager.get_kb_by_name(self._kb_name())

    async def _bind_session_to_kb(
        self,
        event: AstrMessageEvent,
        kb_id: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        current = await sp.session_get(
            event.unified_msg_origin,
            "kb_config",
            default={},
        )
        current = current or {}
        kb_ids = [item for item in current.get("kb_ids", []) if item]
        if kb_id not in kb_ids:
            kb_ids.append(kb_id)

        config = {
            "kb_ids": kb_ids,
            "top_k": self._normalize_top_k(
                top_k if top_k is not None else current.get("top_k")
            ),
        }
        await sp.session_put(event.unified_msg_origin, "kb_config", config)
        return config

    async def _unbind_session_kb(
        self,
        event: AstrMessageEvent,
        kb_id: str,
    ) -> dict[str, Any]:
        current = await sp.session_get(
            event.unified_msg_origin,
            "kb_config",
            default={},
        )
        current = current or {}
        kb_ids = [
            item for item in current.get("kb_ids", []) if item and item != kb_id
        ]
        config = {
            "kb_ids": kb_ids,
            "top_k": self._normalize_top_k(current.get("top_k")),
        }
        await sp.session_put(event.unified_msg_origin, "kb_config", config)
        return config

    async def _session_has_touhou_kb(self, event: AstrMessageEvent) -> bool:
        kb_helper = await self._get_touhou_kb()
        if not kb_helper:
            return False
        current = await sp.session_get(
            event.unified_msg_origin,
            "kb_config",
            default={},
        )
        kb_ids = (current or {}).get("kb_ids", [])
        return kb_helper.kb.kb_id in kb_ids

    async def _get_current_conversation_persona_id(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest | None = None,
    ) -> str | None:
        if request and request.conversation:
            return request.conversation.persona_id

        curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        if not curr_cid:
            return None
        conversation = await self.context.conversation_manager.get_conversation(
            event.unified_msg_origin,
            curr_cid,
        )
        if not conversation:
            return None
        return conversation.persona_id

    async def _resolve_selected_persona_id(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest | None = None,
    ) -> str | None:
        conversation_persona_id = await self._get_current_conversation_persona_id(
            event,
            request,
        )
        provider_settings = (
            self.context.get_config(umo=event.unified_msg_origin).get(
                "provider_settings",
                {},
            )
            or {}
        )
        persona_id, _, _, use_webchat_special_default = (
            await self.context.persona_manager.resolve_selected_persona(
                umo=event.unified_msg_origin,
                conversation_persona_id=conversation_persona_id,
                platform_name=event.get_platform_name(),
                provider_settings=provider_settings,
            )
        )
        if use_webchat_special_default:
            return "_chatui_default_"
        return persona_id

    async def _resolve_touhou_binding_state(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest | None = None,
    ) -> dict[str, Any]:
        persona_id = await self._resolve_selected_persona_id(event, request)
        bound_persona_ids = await self._get_bound_persona_ids()
        session_bound = await self._session_has_touhou_kb(event)
        persona_bound = bool(persona_id and persona_id in bound_persona_ids)
        active_source = ""
        if session_bound:
            active_source = "session"
        elif persona_bound:
            active_source = "persona"

        return {
            "persona_id": persona_id,
            "bound_persona_ids": bound_persona_ids,
            "session_bound": session_bound,
            "persona_bound": persona_bound,
            "active_source": active_source,
        }

    async def _retrieve_touhou_context(
        self,
        query: str,
        kb_name: str,
        umo: str,
        top_k: int | None = None,
        top_k_fusion: int | None = None,
    ) -> dict[str, Any] | None:
        if not query.strip():
            return None
        app_config = self.context.get_config(umo=umo)
        fusion_k = top_k_fusion if top_k_fusion is not None else app_config.get(
            "kb_fusion_top_k",
            20,
        )
        final_k = self._normalize_top_k(top_k)
        return await self.context.kb_manager.retrieve(
            query=query,
            kb_names=[kb_name],
            top_k_fusion=max(1, int(fusion_k)),
            top_m_final=final_k,
        )

    def _build_touhou_hint(self, active_source: str, persona_id: str | None) -> str:
        if active_source == "persona" and persona_id:
            prefix = (
                f"当前生效人格 {persona_id} 已绑定东方Project知识库，"
                "本轮请求需要优先参考该知识库。"
            )
        else:
            prefix = "当前会话已经挂载东方Project知识库。"
        return prefix + TOUHOU_LLM_HINT

    async def _load_existing_doc_map(self, kb_helper: KBHelper) -> dict[str, str]:
        existing: dict[str, str] = {}
        offset = 0
        limit = 500
        while True:
            docs = await kb_helper.list_documents(offset=offset, limit=limit)
            if not docs:
                break
            for doc in docs:
                existing[doc.doc_name] = doc.doc_id
            if len(docs) < limit:
                break
            offset += limit
        return existing

    def _normalize_entry(self, entry: str | None) -> str:
        raw = (entry or "").strip()
        if not raw:
            raw = self._cfg_str("default_entry_page", "东方Project")
        if raw.startswith(("http://", "https://")):
            url = raw
        else:
            encoded = quote(raw.lstrip("/"), safe="/()（）-_")
            url = urljoin(self._cfg_str("base_url", DEFAULT_BASE_URL), encoded)
        parsed = urlparse(url)
        if parsed.netloc not in ALLOWED_HOSTS:
            raise ValueError("只支持 thbwiki.cc / www.thbwiki.cc 下的页面。")
        return parsed._replace(query="", fragment="").geturl()

    def _is_supported_article_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc not in ALLOWED_HOSTS:
            return False
        path = unquote(parsed.path.strip("/"))
        if not path:
            return False
        if path.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".svg",
                ".webp",
                ".pdf",
                ".zip",
                ".mp3",
                ".ogg",
            )
        ):
            return False
        namespace_prefixes = (
            "Special:",
            "THBWiki:",
            "分类:",
            "文件:",
            "模板:",
            "帮助:",
            "用户:",
            "User:",
            "讨论:",
            "MediaWiki:",
        )
        if path.startswith(namespace_prefixes):
            return False
        if ":" in path:
            return False
        return True

    async def _fetch_html(self, url: str) -> str:
        headers = {"User-Agent": self._cfg_str("http_user_agent", "")}
        timeout = self._cfg_float("request_timeout_sec", 20.0)
        proxy = self._cfg_str("http_proxy", "")
        client_kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": timeout,
            "follow_redirects": True,
            "trust_env": True,
        }
        if proxy:
            client_kwargs["proxy"] = proxy
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(url)
            response.raise_for_status()
            final_host = (response.url.host or "").lower()
            if final_host not in ALLOWED_HOSTS:
                raise ValueError(
                    f"页面重定向到了未允许的域名：{final_host or 'unknown'}"
                )
            return response.text

    def _clean_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _truncate_injected_context(self, text: str) -> str:
        cleaned = self._clean_text(text)
        if len(cleaned) <= MAX_CONTEXT_INJECTION_CHARS:
            return cleaned
        truncated = cleaned[:MAX_CONTEXT_INJECTION_CHARS]
        split_at = max(
            truncated.rfind("\n\n"),
            truncated.rfind("\n"),
            truncated.rfind("。"),
            truncated.rfind("！"),
            truncated.rfind("？"),
        )
        if split_at >= int(MAX_CONTEXT_INJECTION_CHARS * 0.6):
            truncated = truncated[: split_at + 1]
        return truncated.rstrip() + "\n[内容已截断]"

    def _extract_categories(self, soup: BeautifulSoup) -> list[str]:
        categories: list[str] = []
        catlinks = soup.select(".catlinks a")
        for link in catlinks:
            name = self._clean_text(link.get_text(" ", strip=True))
            if name and name != "分类" and name not in categories:
                categories.append(name)
        return categories

    def _extract_table_text(self, table: Tag) -> str:
        lines: list[str] = []
        for row in table.select("tr"):
            cells = [
                self._clean_text(cell.get_text(" ", strip=True))
                for cell in row.find_all(["th", "td"], recursive=False)
            ]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            if len(cells) == 1:
                lines.append(cells[0])
            else:
                lines.append(f"{cells[0]}: {' / '.join(cells[1:])}")
        return "\n".join(lines).strip()

    def _extract_block_text(self, tag: Tag) -> str:
        if tag.name in {"ul", "ol"}:
            items = []
            for item in tag.find_all("li", recursive=False):
                text = self._clean_text(item.get_text(" ", strip=True))
                if text:
                    items.append(f"- {text}")
            return "\n".join(items)
        if tag.name == "table":
            return self._extract_table_text(tag)
        text = self._clean_text(tag.get_text(" ", strip=True))
        return text

    def _hard_split_text(self, text: str, limit: int, overlap: int) -> list[str]:
        chunks: list[str] = []
        start = 0
        text = text.strip()
        if not text:
            return chunks
        while start < len(text):
            end = min(len(text), start + limit)
            if end < len(text):
                split_points = [
                    text.rfind("\n", start, end),
                    text.rfind("。", start, end),
                    text.rfind("！", start, end),
                    text.rfind("？", start, end),
                    text.rfind("；", start, end),
                ]
                best = max(split_points)
                if best > start + int(limit * 0.6):
                    end = best + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _build_chunks(
        self,
        title: str,
        url: str,
        categories: list[str],
        sections: list[tuple[str, str]],
    ) -> list[str]:
        max_chars = max(self._cfg_int("pre_chunk_max_chars", 1200), 300)
        overlap = max(self._cfg_int("pre_chunk_overlap", 120), 0)
        prefix_lines = [f"词条：{title}", f"来源：{url}"]
        if categories:
            prefix_lines.append(f"分类：{'、'.join(categories)}")
        prefix = "\n".join(prefix_lines)

        chunks: list[str] = []
        for heading, body in sections:
            text = self._clean_text(body)
            if not text:
                continue
            section_prefix = f"{prefix}\n章节：{heading}\n\n"
            available = max_chars - len(section_prefix)
            if available < 200:
                available = max_chars
                section_prefix = f"{prefix}\n\n"
            parts = self._hard_split_text(text, available, overlap)
            if not parts:
                continue
            for part in parts:
                chunk = f"{section_prefix}{part}".strip()
                chunks.append(chunk)
        return chunks

    def _parse_page(self, url: str, html: str) -> ParsedPage:
        soup = BeautifulSoup(html, "lxml")
        title_node = soup.select_one("#firstHeading")
        title = self._clean_text(
            title_node.get_text(" ", strip=True)
            if title_node
            else unquote(urlparse(url).path.strip("/")) or "未命名词条"
        )
        container = soup.select_one(".mw-parser-output") or soup.select_one(
            "#mw-content-text"
        )
        if not container:
            raise ValueError(f"无法从页面中提取正文: {url}")

        working = BeautifulSoup(str(container), "lxml")
        root = working.select_one(".mw-parser-output") or working
        for selector in REMOVE_SELECTORS:
            for node in root.select(selector):
                node.decompose()

        categories = self._extract_categories(soup)
        links: list[str] = []
        seen_links: set[str] = set()
        for link in root.select("a[href]"):
            href = (link.get("href") or "").strip()
            if not href or href.startswith("#"):
                continue
            candidate = urljoin(url, href)
            candidate = urlparse(candidate)._replace(query="", fragment="").geturl()
            if not self._is_supported_article_url(candidate):
                continue
            if candidate not in seen_links:
                seen_links.add(candidate)
                links.append(candidate)

        sections: list[tuple[str, str]] = []
        current_heading = "概述"
        current_lines: list[str] = []
        for child in root.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            if child.name in {"h2", "h3", "h4"}:
                heading = self._clean_text(child.get_text(" ", strip=True))
                if heading in STOP_SECTION_TITLES:
                    break
                body = self._clean_text("\n\n".join(current_lines))
                if body:
                    sections.append((current_heading, body))
                current_heading = heading or current_heading
                current_lines = []
                continue
            block_text = self._extract_block_text(child)
            if block_text:
                current_lines.append(block_text)

        body = self._clean_text("\n\n".join(current_lines))
        if body:
            sections.append((current_heading, body))

        chunks = self._build_chunks(title, url, categories, sections)
        if not chunks:
            fallback = self._clean_text(root.get_text("\n", strip=True))
            chunks = self._build_chunks(title, url, categories, [("正文", fallback)])
        if not chunks:
            raise ValueError(f"页面正文为空，无法导入: {title}")

        return ParsedPage(
            url=url,
            title=title,
            categories=categories,
            chunks=chunks,
            links=links,
        )

    async def _upsert_page(
        self,
        kb_helper: KBHelper,
        page: ParsedPage,
        existing_docs: dict[str, str],
    ) -> bool:
        async with self._kb_write_lock:
            old_doc_id = existing_docs.get(page.title)
            updated = False
            if old_doc_id:
                await kb_helper.delete_document(old_doc_id)
                updated = True

            doc = await kb_helper.upload_document(
                file_name=page.title,
                file_content=None,
                file_type="txt",
                batch_size=max(1, self._cfg_int("upload_batch_size", 4)),
                tasks_limit=max(1, self._cfg_int("upload_tasks_limit", 1)),
                max_retries=3,
                pre_chunked_text=page.chunks,
            )
            existing_docs[page.title] = doc.doc_id
            return updated

    async def _maybe_auto_bind(
        self,
        event: AstrMessageEvent,
        kb_helper: KBHelper,
    ) -> None:
        if self._cfg_bool("auto_bind_after_init", False):
            await self._bind_session_to_kb(
                event,
                kb_helper.kb.kb_id,
                self._normalize_top_k(None),
            )

    def _task_summary(self, task: SyncTask) -> str:
        return (
            f"任务ID：{task.task_id}\n"
            f"状态：{task.status}\n"
            f"入口：{task.entry}\n"
            f"知识库：{task.kb_name}\n"
            f"已新增：{task.imported}\n"
            f"已更新：{task.updated}\n"
            f"失败：{task.failed}\n"
            f"已发现：{task.discovered}\n"
            f"当前页面：{task.current_title or '-'}\n"
            f"当前URL：{task.current_url or '-'}\n"
            f"开始时间：{self._now_text(task.started_at)}\n"
            f"结束时间：{self._now_text(task.finished_at)}\n"
            f"说明：{task.message or '-'}"
        )

    def _has_running_sync_task(self) -> bool:
        return any(not job.done() for job in self.background_jobs.values())

    def _get_running_sync_task(self) -> SyncTask | None:
        for task_id, job in self.background_jobs.items():
            if not job.done():
                return self.sync_tasks.get(task_id)
        return None

    def _start_sync_task(
        self,
        *,
        entry: str,
        limit: int,
        event_umo: str | None = None,
        auto_bind_session: bool = False,
    ) -> SyncTask:
        running_task = self._get_running_sync_task()
        if running_task:
            raise RuntimeError(
                f"已有运行中的同步任务：{running_task.task_id}。请等待完成后再启动新的同步。"
            )
        limit = max(1, min(limit, 500))
        task_id = uuid.uuid4().hex[:8]
        task = SyncTask(
            task_id=task_id,
            entry=entry or self._default_sync_entry(),
            limit=limit,
            kb_name=self._kb_name(),
            message="任务已创建，等待抓取。",
        )
        self.sync_tasks[task_id] = task
        self.global_last_task_id = task_id
        if event_umo:
            self.session_last_task[event_umo] = task_id

        logger.info(
            "东方知识库同步任务已创建: task_id=%s, entry=%s, limit=%s, kb=%s, session=%s",
            task_id,
            task.entry,
            task.limit,
            task.kb_name,
            event_umo or "-",
        )

        job = asyncio.create_task(
            self._run_sync_task(
                task,
                event_umo=event_umo,
                auto_bind_session=auto_bind_session,
            ),
            name=f"touhou-kb-sync-{task_id}",
        )
        self.background_jobs[task_id] = job

        def _cleanup(_: asyncio.Task) -> None:
            self.background_jobs.pop(task_id, None)

        job.add_done_callback(_cleanup)
        return task

    async def _run_sync_task(
        self,
        task: SyncTask,
        event_umo: str | None = None,
        auto_bind_session: bool = False,
        preferred_provider_id: str = "",
    ) -> None:
        try:
            kb_helper, _ = await self._ensure_kb(preferred_provider_id)
            existing_docs = await self._load_existing_doc_map(kb_helper)
            queue = deque([self._normalize_entry(task.entry)])
            queued = set(queue)
            visited: set[str] = set()
            delay = max(self._cfg_int("crawl_delay_ms", 400), 0) / 1000
            task.status = "running"
            logger.info(
                "东方知识库后台同步开始: task_id=%s, entry=%s, limit=%s, kb=%s",
                task.task_id,
                task.entry,
                task.limit,
                kb_helper.kb.kb_name,
            )

            while queue and (task.imported + task.updated + task.failed) < task.limit:
                current_url = queue.popleft()
                queued.discard(current_url)
                if current_url in visited:
                    continue
                visited.add(current_url)
                task.discovered = len(visited) + len(queue)
                task.current_url = current_url
                logger.info(
                    "东方知识库同步抓取页面: task_id=%s, processed=%s, limit=%s, url=%s",
                    task.task_id,
                    task.imported + task.updated + task.failed,
                    task.limit,
                    current_url,
                )

                try:
                    html = await self._fetch_html(current_url)
                    page = await asyncio.to_thread(self._parse_page, current_url, html)
                    task.current_title = page.title
                    skip_reason = self._get_page_skip_reason(page)
                    if skip_reason:
                        logger.info(
                            "东方知识库同步跳过页面: task_id=%s, title=%s, reason=%s, url=%s",
                            task.task_id,
                            page.title,
                            skip_reason,
                            current_url,
                        )
                        continue
                    was_updated = await self._upsert_page(kb_helper, page, existing_docs)
                    if was_updated:
                        task.updated += 1
                    else:
                        task.imported += 1
                    logger.info(
                        "东方知识库同步写入页面: task_id=%s, action=%s, title=%s, chunks=%s, imported=%s, updated=%s, failed=%s, discovered=%s",
                        task.task_id,
                        "updated" if was_updated else "imported",
                        page.title,
                        len(page.chunks),
                        task.imported,
                        task.updated,
                        task.failed,
                        task.discovered,
                    )

                    for link in page.links:
                        if (
                            link not in visited
                            and link not in queued
                            and len(visited) + len(queue) < task.limit * 4
                        ):
                            queue.append(link)
                            queued.add(link)
                except asyncio.CancelledError:
                    task.errors.append("同步任务已取消。")
                    task.finish("cancelled", "同步任务已取消。")
                    logger.info(
                        "东方知识库后台同步取消: task_id=%s, imported=%s, updated=%s, failed=%s, discovered=%s",
                        task.task_id,
                        task.imported,
                        task.updated,
                        task.failed,
                        task.discovered,
                    )
                    raise
                except Exception as exc:
                    task.failed += 1
                    detail = f"{current_url} -> {type(exc).__name__}: {exc}"
                    task.errors.append(detail)
                    logger.warning(
                        "东方知识库同步失败: task_id=%s, detail=%s",
                        task.task_id,
                        detail,
                    )

                if delay > 0:
                    await asyncio.sleep(delay)

            message = (
                f"同步完成，新增 {task.imported} 页，更新 {task.updated} 页，失败 {task.failed} 页。"
            )
            task.finish("completed", message)
            logger.info(
                "东方知识库后台同步完成: task_id=%s, status=%s, imported=%s, updated=%s, failed=%s, discovered=%s, kb=%s",
                task.task_id,
                task.status,
                task.imported,
                task.updated,
                task.failed,
                task.discovered,
                kb_helper.kb.kb_name,
            )

            if auto_bind_session and event_umo and self._cfg_bool(
                "auto_bind_after_init",
                False,
            ):
                current = await sp.session_get(event_umo, "kb_config", default={})
                current = current or {}
                kb_ids = [item for item in current.get("kb_ids", []) if item]
                if kb_helper.kb.kb_id not in kb_ids:
                    kb_ids.append(kb_helper.kb.kb_id)
                await sp.session_put(
                    event_umo,
                    "kb_config",
                    {
                        "kb_ids": kb_ids,
                        "top_k": self._normalize_top_k(current.get("top_k")),
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"东方知识库后台同步失败: {exc}")
            logger.error(traceback.format_exc())
            task.errors.append(str(exc))
            task.finish("failed", f"同步失败: {exc}")

    @filter.on_llm_request()
    async def inject_touhou_hint(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
    ) -> None:
        query = self._sanitize_kb_query(request.prompt or event.message_str or "")
        if not self._cfg_bool("enable_routing_hint", True):
            return
        if not self._is_touhou_intent(query):
            return

        kb_helper = await self._get_touhou_kb()
        if not kb_helper:
            return

        binding_state = await self._resolve_touhou_binding_state(event, request)
        active_source = binding_state["active_source"]
        if not active_source:
            return

        system_prompt = request.system_prompt or ""
        hint = self._build_touhou_hint(
            active_source,
            binding_state["persona_id"],
        )
        if hint not in system_prompt:
            request.system_prompt = f"{system_prompt}\n{hint}\n"

        try:
            kb_result = await self._retrieve_touhou_context(
                query=query,
                kb_name=kb_helper.kb.kb_name,
                umo=event.unified_msg_origin,
            )
            context_text = self._truncate_injected_context(
                (kb_result or {}).get("context_text", "")
            )
            if context_text and context_text not in request.system_prompt:
                request.system_prompt += (
                    f"\n\n[Related Knowledge Base Results]:\n{context_text}"
                )
                logger.info(
                    "东方知识库已注入检索结果: source=%s, persona=%s, kb=%s",
                    active_source,
                    binding_state["persona_id"],
                    kb_helper.kb.kb_name,
                )
        except Exception as exc:
            logger.warning(f"注入东方知识库检索结果失败: {exc}")
        if self._cfg_bool("remove_competing_tools", True):
            self._remove_competing_tools(request)

    @filter.command("东方知识库初始化")
    async def initialize_touhou_kb(
        self,
        event: AstrMessageEvent,
        embedding_provider_id: str = "",
    ):
        try:
            kb_helper, created = await self._ensure_kb(embedding_provider_id)
            await self._maybe_auto_bind(event, kb_helper)
            provider_id = kb_helper.kb.embedding_provider_id or "-"
            rerank_provider_id = getattr(kb_helper.kb, "rerank_provider_id", None) or "-"
            status = "已创建" if created else "已复用"
            lines = [
                f"{status}知识库：{kb_helper.kb.kb_name}",
                f"kb_id：{kb_helper.kb.kb_id}",
                f"Embedding Provider：{provider_id}",
                f"Rerank Provider：{rerank_provider_id}",
                f"文档数：{kb_helper.kb.doc_count}，块数：{kb_helper.kb.chunk_count}",
            ]

            if (
                self._cfg_bool("auto_sync_after_init", True)
                and kb_helper.kb.doc_count == 0
                and kb_helper.kb.chunk_count == 0
                and not self._has_running_sync_task()
            ):
                try:
                    task = self._start_sync_task(
                        entry=self._default_sync_entry(),
                        limit=self._default_sync_limit(),
                        event_umo=event.unified_msg_origin,
                        auto_bind_session=False,
                    )
                    logger.info(
                        "东方知识库初始化后自动启动同步: task_id=%s, entry=%s, limit=%s, kb=%s",
                        task.task_id,
                        task.entry,
                        task.limit,
                        kb_helper.kb.kb_name,
                    )
                    lines.extend(
                        [
                            "",
                            "后台同步已自动启动：",
                            f"任务ID：{task.task_id}",
                            f"入口：{task.entry}",
                            f"页数上限：{task.limit}",
                            "可用 /东方知识库状态 或 /东方知识库任务 查看进度。",
                        ]
                    )
                except RuntimeError as exc:
                    lines.extend(["", f"后台同步未启动：{exc}"])
            yield event.plain_result("\n".join(lines))
        except Exception as exc:
            yield event.plain_result(f"初始化失败：{exc}")

    @filter.command("东方知识库绑定")
    async def bind_touhou_kb(
        self,
        event: AstrMessageEvent,
        top_k: int = 5,
    ):
        kb_helper = await self._get_touhou_kb()
        if not kb_helper:
            yield event.plain_result("知识库还不存在，请先执行 /东方知识库初始化")
            return
        normalized_top_k = self._normalize_top_k(top_k)
        config = await self._bind_session_to_kb(
            event,
            kb_helper.kb.kb_id,
            normalized_top_k,
        )
        yield event.plain_result(
            f"已把当前会话绑定到 {kb_helper.kb.kb_name}\n"
            f"kb_ids：{', '.join(config.get('kb_ids', []))}\n"
            f"top_k：{config.get('top_k', normalized_top_k)}"
        )

    @filter.command("东方知识库绑定人格")
    async def bind_touhou_persona(
        self,
        event: AstrMessageEvent,
        persona_id: str = "",
    ):
        kb_helper = await self._get_touhou_kb()
        if not kb_helper:
            yield event.plain_result("知识库还不存在，请先执行 /东方知识库初始化")
            return
        try:
            target_persona_id = persona_id.strip()
            if not target_persona_id:
                target_persona_id = await self._resolve_selected_persona_id(event)
            target_persona_id = self._normalize_persona_id(target_persona_id)
            bindings = await self._bind_persona_id(target_persona_id)
            yield event.plain_result(
                f"已把人格 {target_persona_id} 绑定到 {kb_helper.kb.kb_name}\n"
                f"当前已绑定人格：{bindings or []}"
            )
        except Exception as exc:
            yield event.plain_result(f"绑定人格失败：{exc}")

    @filter.command("东方知识库解绑人格")
    async def unbind_touhou_persona(
        self,
        event: AstrMessageEvent,
        persona_id: str = "",
    ):
        try:
            target_persona_id = persona_id.strip()
            if not target_persona_id:
                target_persona_id = await self._resolve_selected_persona_id(event)
            target_persona_id = self._normalize_persona_id(target_persona_id)
            bindings = await self._unbind_persona_id(target_persona_id)
            yield event.plain_result(
                f"已解除人格 {target_persona_id} 的东方知识库绑定\n"
                f"当前已绑定人格：{bindings or []}"
            )
        except Exception as exc:
            yield event.plain_result(f"解绑人格失败：{exc}")

    @filter.command("东方知识库人格列表")
    async def list_touhou_persona_bindings(self, event: AstrMessageEvent):
        current_persona_id = await self._resolve_selected_persona_id(event)
        bindings = await self._get_bound_persona_ids()
        lines = [
            f"当前生效人格：{current_persona_id or '-'}",
            f"已绑定人格：{bindings or []}",
            "说明：命中这些人格时，即使当前会话没绑定知识库，也会自动检索东方知识库。",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("东方知识库解绑")
    async def unbind_touhou_kb(self, event: AstrMessageEvent):
        kb_helper = await self._get_touhou_kb()
        if not kb_helper:
            yield event.plain_result("东方知识库不存在，无需解绑。")
            return
        config = await self._unbind_session_kb(event, kb_helper.kb.kb_id)
        if config.get("kb_ids"):
            yield event.plain_result(
                "已从当前会话移除东方知识库，其它知识库绑定保持不变。"
            )
        else:
            yield event.plain_result("已从当前会话移除东方知识库，当前会话不再使用知识库。")

    @filter.command("东方知识库状态")
    async def touhou_kb_status(self, event: AstrMessageEvent):
        kb_helper = await self._get_touhou_kb()
        kb_config = await sp.session_get(
            event.unified_msg_origin,
            "kb_config",
            default={},
        )
        binding_state = await self._resolve_touhou_binding_state(event)
        last_task_id = self.session_last_task.get(event.unified_msg_origin)
        lines = []
        if kb_helper:
            lines.append(f"知识库：{kb_helper.kb.kb_name}")
            lines.append(f"kb_id：{kb_helper.kb.kb_id}")
            lines.append(f"文档数：{kb_helper.kb.doc_count}")
            lines.append(f"块数：{kb_helper.kb.chunk_count}")
            lines.append(
                f"Embedding Provider：{kb_helper.kb.embedding_provider_id or '-'}"
            )
            lines.append(
                f"Rerank Provider：{getattr(kb_helper.kb, 'rerank_provider_id', None) or '-'}"
            )
        else:
            lines.append("知识库：未初始化")

        session_kb_ids = (kb_config or {}).get("kb_ids", [])
        lines.append(
            "当前会话已绑定："
            + ("是" if kb_helper and kb_helper.kb.kb_id in session_kb_ids else "否")
        )
        lines.append(f"当前会话 kb_ids：{session_kb_ids or []}")
        lines.append(
            f"当前会话 top_k：{self._normalize_top_k((kb_config or {}).get('top_k'))}"
        )
        lines.append(f"当前生效人格：{binding_state['persona_id'] or '-'}")
        lines.append("当前人格已绑定：" + ("是" if binding_state["persona_bound"] else "否"))
        lines.append(f"已绑定人格列表：{binding_state['bound_persona_ids'] or []}")
        lines.append(f"当前路由来源：{binding_state['active_source'] or '-'}")

        if last_task_id and last_task_id in self.sync_tasks:
            task = self.sync_tasks[last_task_id]
            lines.append("")
            lines.append("最近任务：")
            lines.append(self._task_summary(task))
        elif self.global_last_task_id and self.global_last_task_id in self.sync_tasks:
            task = self.sync_tasks[self.global_last_task_id]
            lines.append("")
            lines.append("最近全局任务：")
            lines.append(self._task_summary(task))

        yield event.plain_result("\n".join(lines))

    @filter.command("东方知识库导入")
    async def import_single_page(
        self,
        event: AstrMessageEvent,
        page: str = "东方Project",
    ):
        running_task = self._get_running_sync_task()
        if running_task:
            yield event.plain_result(
                f"当前有同步任务正在运行：{running_task.task_id}\n请等待同步完成后再执行单页导入。"
            )
            return
        try:
            kb_helper, _ = await self._ensure_kb()
            url = self._normalize_entry(page)
            html = await self._fetch_html(url)
            parsed = await asyncio.to_thread(self._parse_page, url, html)
            skip_reason = self._get_page_skip_reason(parsed)
            if skip_reason:
                yield event.plain_result(
                    f"已跳过：{parsed.title}\n"
                    f"原因：{skip_reason}\n"
                    f"URL：{parsed.url}"
                )
                return
            existing_docs = await self._load_existing_doc_map(kb_helper)
            was_updated = await self._upsert_page(kb_helper, parsed, existing_docs)
            await self._maybe_auto_bind(event, kb_helper)
            action = "更新" if was_updated else "导入"
            logger.info(
                "东方知识库单页导入完成: action=%s, title=%s, url=%s, chunks=%s, kb=%s",
                action,
                parsed.title,
                parsed.url,
                len(parsed.chunks),
                kb_helper.kb.kb_name,
            )
            yield event.plain_result(
                f"{action}完成：{parsed.title}\n"
                f"URL：{parsed.url}\n"
                f"分类：{'、'.join(parsed.categories) if parsed.categories else '-'}\n"
                f"写入块数：{len(parsed.chunks)}"
            )
        except Exception as exc:
            yield event.plain_result(f"导入失败：{exc}")

    @filter.command("东方知识库同步")
    async def sync_touhou_pages(
        self,
        event: AstrMessageEvent,
        entry: str = "东方Project",
        limit: int = 40,
    ):
        try:
            task = self._start_sync_task(
                entry=entry or self._default_sync_entry(),
                limit=limit,
                event_umo=event.unified_msg_origin,
                auto_bind_session=False,
            )
        except RuntimeError as exc:
            yield event.plain_result(str(exc))
            return

        yield event.plain_result(
            f"已开始同步任务：{task.task_id}\n"
            f"入口：{task.entry}\n"
            f"页数上限：{task.limit}\n"
            "可用 /东方知识库任务 查看进度。"
        )

    async def terminate(self):
        running_jobs = list(self.background_jobs.values())
        for job in running_jobs:
            if not job.done():
                job.cancel()
        if running_jobs:
            await asyncio.gather(*running_jobs, return_exceptions=True)
        self.background_jobs.clear()

    @filter.command("东方知识库任务")
    async def touhou_kb_task_status(
        self,
        event: AstrMessageEvent,
        task_id: str = "",
    ):
        target_task_id = task_id.strip() or self.session_last_task.get(
            event.unified_msg_origin,
            "",
        )
        if not target_task_id:
            target_task_id = self.global_last_task_id
        if not target_task_id:
            yield event.plain_result("目前还没有同步任务记录。")
            return
        task = self.sync_tasks.get(target_task_id)
        if not task:
            yield event.plain_result(f"找不到任务：{target_task_id}")
            return
        lines = [self._task_summary(task)]
        if task.errors:
            lines.append("")
            lines.append("最近错误：")
            lines.extend(task.errors[-5:])
        yield event.plain_result("\n".join(lines))

    @filter.command("东方知识库搜索")
    async def search_touhou_kb(
        self,
        event: AstrMessageEvent,
        query: GreedyStr,
    ):
        kb_helper = await self._get_touhou_kb()
        if not kb_helper:
            yield event.plain_result("知识库还不存在，请先执行 /东方知识库初始化")
            return
        result = await self.context.kb_manager.retrieve(
            query=query,
            kb_names=[kb_helper.kb.kb_name],
            top_k_fusion=10,
            top_m_final=5,
        )
        if not result or not result.get("results"):
            yield event.plain_result(f"没有检索到与“{query}”相关的知识块。")
            return

        lines = [f"检索词：{query}"]
        for idx, item in enumerate(result["results"][:5], start=1):
            excerpt = self._clean_text(item.get("content", ""))[:120]
            lines.append(
                f"{idx}. [{item.get('doc_name', '-')}] "
                f"score={item.get('score', 0):.4f} "
                f"{excerpt}"
            )
        yield event.plain_result("\n".join(lines))
