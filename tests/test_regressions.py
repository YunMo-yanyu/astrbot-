"""Offline regression tests; AstrBot services are replaced only at the import boundary."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch


def load_plugin():
    modules = {}
    exports = {
        'astrbot.api': ['AstrBotConfig', 'logger', 'sp'],
        'astrbot.api.event': ['AstrMessageEvent', 'filter'],
        'astrbot.api.provider': ['ProviderRequest'],
        'astrbot.core.knowledge_base.kb_helper': ['KBHelper'],
        'astrbot.core.provider.provider': ['EmbeddingProvider', 'RerankProvider'],
        'astrbot.core.star': ['Context', 'Star'],
        'astrbot.core.star.filter.command': ['GreedyStr'],
    }
    for name, attrs in exports.items():
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        modules[name] = module
    class Filters:
        PermissionType = types.SimpleNamespace(ADMIN="admin")
        def __getattr__(self, name):
            return lambda *a, **kw: lambda fn: fn
    modules['astrbot.api.event'].filter = Filters()
    spec = importlib.util.spec_from_file_location('plugin_under_test', Path(__file__).parents[1] / 'main.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**modules, spec.name: module}):
        spec.loader.exec_module(module)
    return module


plugin = load_plugin()


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.main = object.__new__(plugin.Main)
        self.main.config = {}
        self.main._kb_write_lock = asyncio.Lock()
        self.main._policy_store_lock = asyncio.Lock()
        self.records = {}
        async def get(key, default=None):
            return self.records.get(key, default)
        async def put(key, value):
            self.records[key] = value
        self.main.get_kv_data = AsyncMock(side_effect=get)
        self.main.put_kv_data = AsyncMock(side_effect=put)

    def parse(self, body):
        return self.main._parse_page('https://thbwiki.cc/博丽灵梦',
            '<h1 id="firstHeading">博丽灵梦</h1><div class="mw-parser-output">' + body + '</div>')

    def test_sections_after_notes_are_preserved(self):
        page = self.parse('<p>开头</p><h2>注释</h2><p>应去掉</p><h3>注释子节</h3><p>也去掉</p>'
                          '<h2>角色设定</h2><p>后续有效正文</p>')
        text = '\n'.join(page.chunks)
        self.assertIn('开头', text)
        self.assertIn('后续有效正文', text)
        self.assertNotIn('应去掉', text)
        self.assertNotIn('也去掉', text)

    def test_wrapped_headings_bare_text_and_template_content(self):
        page = self.parse('<div><div class="mw-heading"><h2>能力</h2></div>裸文本'
                          '<p>能力描述</p><div class="noprint">模板中的有效正文</div></div>'
                          '<div class="navbox">导航噪音</div>')
        text = '\n'.join(page.chunks)
        for expected in ['章节：能力', '裸文本', '能力描述', '模板中的有效正文']:
            self.assertIn(expected, text)
        self.assertNotIn('导航噪音', text)

    def test_empty_filtered_page_does_not_reintroduce_notes(self):
        with self.assertRaisesRegex(ValueError, '正文为空'):
            self.parse('<h2>参考资料</h2><p>只有参考资料</p>')

    def test_chunks_cover_every_character_without_excessive_overlap(self):
        text = ''.join(chr(0x4e00 + i) for i in range(2000))
        chunks = self.main._hard_split_text(text, 300, 5000)
        self.assertLess(len(chunks), 12)
        for char in text:
            self.assertTrue(any(char in chunk for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 300 for chunk in chunks))

    def test_oversized_page_fails_explicitly_instead_of_truncating(self):
        self.main.config = {'max_page_chunks': 1, 'pre_chunk_max_chars': 300}
        with self.assertRaisesRegex(ValueError, '超过单页上限'):
            self.parse('<p>' + '灵梦' * 1000 + '</p>')

    def test_upload_failure_keeps_old_document(self):
        helper = types.SimpleNamespace(upload_document=AsyncMock(side_effect=RuntimeError('embedding failed')),
                                       delete_document=AsyncMock())
        docs = {'灵梦': 'old'}
        page = plugin.ParsedPage('url', '灵梦', [], ['正文'], [])
        with self.assertRaises(RuntimeError):
            asyncio.run(self.main._upsert_page(helper, page, docs))
        helper.delete_document.assert_not_awaited()
        self.assertEqual(docs, {'灵梦': 'old'})

    def test_successful_upload_precedes_old_deletion(self):
        events = []
        async def upload(**kw):
            events.append('upload')
            return types.SimpleNamespace(doc_id='new')
        async def delete(doc_id):
            events.append('delete:' + doc_id)
        helper = types.SimpleNamespace(upload_document=upload, delete_document=delete)
        docs = {'灵梦': 'old'}
        page = plugin.ParsedPage('url', '灵梦', [], ['正文'], [])
        self.assertTrue(asyncio.run(self.main._upsert_page(helper, page, docs)))
        self.assertEqual(events, ['upload', 'delete:old'])
        self.assertEqual(docs['灵梦'], 'new')

    def test_delete_failure_preserves_uploaded_version(self):
        helper = types.SimpleNamespace(upload_document=AsyncMock(return_value=types.SimpleNamespace(doc_id='new')),
            delete_document=AsyncMock(side_effect=RuntimeError('delete failed')))
        docs = {'灵梦': 'old'}
        with self.assertRaisesRegex(RuntimeError, '新版本已写入'):
            asyncio.run(self.main._upsert_page(helper, plugin.ParsedPage('url','灵梦',[],['正文'],[]), docs))
        self.assertEqual([call.args[0] for call in helper.delete_document.await_args_list], ['old'])
        self.assertEqual(docs['灵梦'], 'new')

class SyncAndRoutingTests(unittest.TestCase):
    setUp = RegressionTests.setUp
    def test_crawl_budget_counts_filtered_pages(self):
        main = self.main
        main.config = {'crawl_delay_ms': 0}
        helper = types.SimpleNamespace(kb=types.SimpleNamespace(kb_name='test'))
        main._ensure_kb = AsyncMock(return_value=(helper, False))
        main._load_existing_doc_map = AsyncMock(return_value={})
        main._fetch_html = AsyncMock(return_value='html')
        main._upsert_page = AsyncMock(return_value=False)
        links = ['https://thbwiki.cc/skip' + str(i) for i in range(10)]
        def parse(url, html):
            return plugin.ParsedPage(url, 'LostWord' if 'skip' in url else '东方Project', [], ['正文'], links)
        main._parse_page = parse
        task = plugin.SyncTask('test', '东方Project', 3, 'test')
        from unittest.mock import Mock
        with patch.object(plugin, 'logger', Mock()):
            asyncio.run(main._run_sync_task(task))
        self.assertEqual(main._fetch_html.await_count, 3)
        self.assertEqual(task.imported, 1)
        self.assertEqual(task.status, 'completed')

    def test_failed_or_empty_retrieval_keeps_external_tools(self):
        for result in [None, RuntimeError('retrieval failed')]:
            main = self.main
            main._get_touhou_kb = AsyncMock(return_value=types.SimpleNamespace(kb=types.SimpleNamespace(kb_name='test')))
            main._resolve_touhou_binding_state = AsyncMock(return_value={'active_source':'persona','persona_id':'test'})
            main._retrieve_touhou_context = AsyncMock(side_effect=result) if isinstance(result, Exception) else AsyncMock(return_value=result)
            from unittest.mock import Mock
            main._remove_competing_tools = Mock()
            request = types.SimpleNamespace(prompt='灵梦是谁', system_prompt='')
            event = types.SimpleNamespace(message_str='', unified_msg_origin='test')
            with patch.object(plugin, 'logger', Mock()):
                asyncio.run(main.inject_touhou_hint(event, request))
            main._remove_competing_tools.assert_not_called()

    def test_explicit_search_request_keeps_tools_with_context(self):
        main = self.main
        main._get_touhou_kb = AsyncMock(return_value=types.SimpleNamespace(kb=types.SimpleNamespace(kb_name='test')))
        main._resolve_touhou_binding_state = AsyncMock(return_value={'active_source':'persona','persona_id':'test'})
        main._retrieve_touhou_context = AsyncMock(return_value={'context_text':'知识库内容'})
        from unittest.mock import Mock
        main._remove_competing_tools = Mock()
        with patch.object(plugin, 'logger', Mock()):
            asyncio.run(main.inject_touhou_hint(types.SimpleNamespace(message_str='',unified_msg_origin='test'),
                types.SimpleNamespace(prompt='联网搜索灵梦的设定', system_prompt='')))
        main._remove_competing_tools.assert_not_called()


if __name__ == '__main__':
    unittest.main()
