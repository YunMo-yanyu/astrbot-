import asyncio
import json
from pathlib import Path
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import test_regressions as regressions

plugin = regressions.plugin


class PolicyTests(unittest.TestCase):
    setUp = regressions.RegressionTests.setUp

    def small_config(self, **overrides):
        self.main.config = {f'core_seed_{t}': '' for t in plugin.DEFAULT_CORE_SEEDS}
        self.main.config.update({'core_seed_character': '博丽灵梦', 'crawl_delay_ms': 0})
        self.main.config.update(overrides)

    def url(self, title):
        return self.main._normalize_entry(title)

    def run_sync(self, pages, entry=plugin.CORE_ENTRY, limit=40):
        main = self.main
        helper = types.SimpleNamespace(kb=types.SimpleNamespace(kb_name='test'))
        main._ensure_kb = AsyncMock(return_value=(helper, False))
        main._load_existing_doc_map = AsyncMock(return_value={})
        main._fetch_html = AsyncMock(return_value='fixture')
        main._parse_page = lambda url, html: pages[url]
        main._upsert_page = AsyncMock(return_value=False)
        task = plugin.SyncTask('test', entry, limit, 'test')
        with patch.object(plugin, 'logger', Mock()):
            asyncio.run(main._run_sync_task(task))
        self.assertEqual(task.status, 'completed', task.errors)
        return task

    def page(self, title, categories=(), links=()):
        return plugin.ParsedPage(self.url(title), title, list(categories), ['正文'], [self.url(t) for t in links])

    def test_schema_defaults_match_policy(self):
        schema = json.loads((Path(__file__).parents[1] / '_conf_schema.json').read_text())
        self.assertEqual(schema['default_entry_page']['default'], self.main._default_sync_entry())
        for t, defaults in plugin.DEFAULT_CORE_SEEDS.items():
            self.assertEqual(schema['core_seed_' + t]['default'].splitlines(), list(defaults))

    def test_seed_groups_dedupe_urls_and_allow_empty_group(self):
        self.small_config(core_seed_character='博丽灵梦\nhttps://www.thbwiki.cc/博丽灵梦#设定', core_seed_work='东方红魔乡')
        self.assertEqual(self.main._core_seed_map(), {'博丽灵梦': 'character', '东方红魔乡': 'work'})
        self.assertNotIn('藤原妹红', self.main._core_seed_map())

    def test_classification_is_conservative_and_derivative_wins(self):
        self.small_config()
        self.assertEqual(self.main._classify_page('博丽灵梦', [])[0], 'character')
        self.assertEqual(self.main._classify_page('博丽灵梦', ['同人角色'])[0], 'derivative')
        self.assertEqual(self.main._classify_page('新地点', ['地点'])[0], 'location')
        self.assertEqual(self.main._classify_page('灵梦同人', [])[0], 'unknown')
        self.assertEqual(self.main._classify_page('博丽灵梦/设定文档', [])[0], 'source')
        self.assertEqual(self.main._classify_page('博丽灵梦/随便什么', [])[0], 'unknown')

    def test_all_seeds_are_requested_before_expansion(self):
        self.small_config(core_seed_work='东方红魔乡')
        pages = {self.url('博丽灵梦'): self.page('博丽灵梦', links=['琪露诺']),
                 self.url('东方红魔乡'): self.page('东方红魔乡'),
                 self.url('琪露诺'): self.page('琪露诺', ['角色'])}
        task = self.run_sync(pages, limit=2)
        self.assertEqual([c.args[0] for c in self.main._fetch_html.await_args_list],
                         [self.url('博丽灵梦'), self.url('东方红魔乡')])
        self.assertEqual(task.imported, 2)
        self.assertIn('预算已用完', task.message)
        records = self.records[plugin.CANDIDATE_STORE_KEY]
        self.assertEqual(records[self.url('琪露诺')]['status'], 'pending')

    def test_only_allowed_types_are_embedded_and_depth_two_stays_pending(self):
        self.small_config()
        pages = {self.url('博丽灵梦'): self.page('博丽灵梦', links=['琪露诺', '不明资料', '同人角色页']),
                 self.url('琪露诺'): self.page('琪露诺', ['角色'], ['更深资料']),
                 self.url('不明资料'): self.page('不明资料'),
                 self.url('同人角色页'): self.page('同人角色页', ['同人角色'])}
        task = self.run_sync(pages)
        self.assertEqual(task.requested, 4)
        self.assertEqual(task.imported, 2)
        self.assertEqual(task.pending, 2)
        self.assertEqual([c.args[1].title for c in self.main._upsert_page.await_args_list], ['博丽灵梦', '琪露诺'])
        records = self.records[plugin.CANDIDATE_STORE_KEY]
        self.assertEqual(records[self.url('不明资料')]['content_type'], 'unknown')
        self.assertEqual(records[self.url('同人角色页')]['content_type'], 'derivative')
        self.assertEqual(records[self.url('更深资料')]['depth'], 2)
        self.assertEqual(records[self.url('更深资料')]['status'], 'pending')

    def test_zero_depth_only_requests_seeds_and_title_exclusions_cost_no_request(self):
        self.small_config(sync_max_depth=0)
        pages = {self.url('博丽灵梦'): self.page('博丽灵梦', links=['琪露诺', 'LostWord'])}
        task = self.run_sync(pages)
        self.assertEqual(task.requested, 1)
        records = self.records[plugin.CANDIDATE_STORE_KEY]
        self.assertEqual(records[self.url('LostWord')]['status'], 'excluded')
        self.assertEqual(records[self.url('琪露诺')]['status'], 'pending')

    def test_manual_entry_is_allowed_even_when_unknown(self):
        self.small_config()
        pages = {self.url('指定资料'): self.page('指定资料')}
        task = self.run_sync(pages, entry='指定资料')
        self.assertEqual(task.imported, 1)

    def test_unknown_can_be_explicitly_enabled_for_expansion(self):
        self.small_config(sync_allowed_types='unknown')
        pages = {self.url('博丽灵梦'): self.page('博丽灵梦', links=['不明资料']),
                 self.url('不明资料'): self.page('不明资料')}
        self.assertEqual(self.run_sync(pages).imported, 2)

    def test_exclusion_overrides_core_seed(self):
        self.small_config(core_seed_character='LostWord')
        task = self.run_sync({self.url('LostWord'): self.page('LostWord')})
        self.assertEqual(task.imported, 0)
        self.assertEqual(task.skipped, 1)
        self.main._upsert_page.assert_not_awaited()

    def test_fetch_or_parse_failure_is_persisted_without_upload(self):
        self.small_config()
        task = self.run_sync({})
        self.assertEqual(task.failed, 1)
        self.assertEqual(task.requested, 1)
        self.main._upsert_page.assert_not_awaited()
        self.assertEqual(self.records[plugin.CANDIDATE_STORE_KEY][self.url('博丽灵梦')]['status'], 'failed')

    def test_equivalent_links_are_canonicalized_and_deduplicated(self):
        html = ('<h1 id="firstHeading">博丽灵梦</h1><div class="mw-parser-output"><p>正文</p>'
                '<a href="/雾雨魔理沙">一</a><a href="https://www.thbwiki.cc/%E9%9B%BE%E9%9B%A8%E9%AD%94%E7%90%86%E6%B2%99#设定">二</a></div>')
        page = self.main._parse_page(self.url('博丽灵梦'), html)
        self.assertEqual(page.links, [self.url('雾雨魔理沙')])

    def test_candidate_store_bounded_persistent_and_does_not_downgrade_imported(self):
        self.small_config(max_candidate_records=20)
        asyncio.run(self.main._save_candidates({str(i): {'status':'pending','updated_at': i} for i in range(30)}))
        self.assertEqual(len(self.records[plugin.CANDIDATE_STORE_KEY]), 20)
        self.assertNotIn('0', self.records[plugin.CANDIDATE_STORE_KEY])
        asyncio.run(self.main._save_candidates({'keep':{'status':'imported','updated_at':31}}))
        asyncio.run(self.main._save_candidates({'keep':{'status':'pending','reason':'待分类／未抓取','updated_at':32}}))
        self.assertEqual(self.records[plugin.CANDIDATE_STORE_KEY]['keep']['status'], 'imported')

    def test_missing_query_counts_are_bounded_and_can_be_disabled(self):
        async def record():
            await self.main._record_missing_query('灵梦能力')
            await self.main._record_missing_query('灵梦能力')
            for i in range(110):
                await self.main._record_missing_query('问题' + str(i))
        asyncio.run(record())
        records = self.records[plugin.MISSING_QUERY_STORE_KEY]
        self.assertEqual(len(records), 100)
        self.assertNotIn('灵梦能力', records)
        self.main.config['record_missing_queries'] = False
        asyncio.run(self.main._record_missing_query('禁止记录'))
        self.assertNotIn('禁止记录', records)

    def test_missing_query_count_and_clear_command(self):
        async def run():
            await self.main._record_missing_query('灵梦能力')
            await self.main._record_missing_query('灵梦能力')
            self.assertEqual(self.records[plugin.MISSING_QUERY_STORE_KEY]['灵梦能力']['count'], 2)
            event = types.SimpleNamespace(plain_result=lambda text: text)
            result = [text async for text in self.main.clear_missing_queries(event)]
            self.assertIn('已清空', result[0])
            self.assertEqual(self.records[plugin.MISSING_QUERY_STORE_KEY], {})
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
