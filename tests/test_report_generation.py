import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ai_aquatica.report_generation import (
    generate_statistical_report,
    generate_interpretation_report,
    suggest_further_analysis,
)

class HTMLNode:
    def __init__(self, tag, attrs, parent=None):
        self.tag = tag
        self.attrs = dict(attrs)
        self.parent = parent
        self.children = []
        self.text = ''

    def add_child(self, child):
        self.children.append(child)

    def get_text(self):
        combined = self.text
        for child in self.children:
            combined += child.get_text()
        return combined

    def has_class(self, class_name):
        classes = self.attrs.get('class', '')
        return class_name in classes.split()


class SimpleHTMLTreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root = HTMLNode('root', {})
        self.current = self.root

    def handle_starttag(self, tag, attrs):
        node = HTMLNode(tag, attrs, parent=self.current)
        self.current.add_child(node)
        self.current = node

    def handle_endtag(self, tag):
        while self.current is not None and self.current.tag != tag:
            self.current = self.current.parent
        if self.current is not None and self.current.parent is not None:
            self.current = self.current.parent
        else:
            self.current = self.root

    def handle_data(self, data):
        if self.current is not None:
            self.current.text += data


def parse_html_tree(path: Path) -> HTMLNode:
    parser = SimpleHTMLTreeBuilder()
    parser.feed(path.read_text(encoding='utf-8'))
    parser.close()
    return parser.root


def find_first(node: HTMLNode, tag: str = None, predicate=None):
    for child in node.children:
        if (tag is None or child.tag == tag) and (predicate is None or predicate(child)):
            return child
        result = find_first(child, tag, predicate)
        if result is not None:
            return result
    return None


def find_all(node: HTMLNode, predicate):
    matches = []
    for child in node.children:
        if predicate(child):
            matches.append(child)
        matches.extend(find_all(child, predicate))
    return matches


def count_descendants(node: HTMLNode, tag: str) -> int:
    return len(find_all(node, lambda n: n.tag == tag))


class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        self.X = self.data[['feature1', 'feature2']]
        self.y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        temp_path = Path(self.temp_dir.name)
        self.reports = {
            'statistical': temp_path / 'statistical_report.html',
            'interpretation': temp_path / 'interpretation_report.html',
            'further': temp_path / 'further_analysis_report.html',
            'heatmap': temp_path / 'heatmap.png',
        }

    def tearDown(self):
        os.chdir(self.original_cwd)
        for report in self.reports.values():
            if report.exists():
                report.unlink()

    def test_generate_statistical_report(self):
        generate_statistical_report(self.data)
        self.assertTrue(self.reports['statistical'].exists())
        self.assertTrue(self.reports['heatmap'].exists())

        root = parse_html_tree(self.reports['statistical'])

        title = find_first(root, tag='h1')
        self.assertIsNotNone(title)
        self.assertEqual(title.get_text().strip(), 'Statistical Report')

        tables = find_all(root, lambda n: n.tag == 'table')
        self.assertGreaterEqual(len(tables), 2)
        self.assertIn('feature1', tables[0].get_text())
        self.assertIn('range', tables[0].get_text())
        self.assertIn('feature2', tables[1].get_text())

        heatmap_img = find_first(
            root,
            tag='img',
            predicate=lambda n: n.attrs.get('alt') == 'Correlation Heatmap',
        )
        self.assertIsNotNone(heatmap_img)
        self.assertEqual(heatmap_img.attrs.get('src'), 'heatmap.png')

    def test_generate_interpretation_report(self):
        generate_interpretation_report(self.data, self.model, self.X_test, self.y_test)
        self.assertTrue(self.reports['interpretation'].exists())

        root = parse_html_tree(self.reports['interpretation'])

        title = find_first(root, tag='h1')
        self.assertIsNotNone(title)
        self.assertEqual(title.get_text().strip(), 'Model Interpretation Report')

        metric_nodes = find_all(root, lambda n: n.has_class('metric'))
        self.assertGreaterEqual(len(metric_nodes), 3)

        metrics = {}
        for node in metric_nodes:
            tokens = node.text.strip().split()
            if not tokens:
                continue
            label = ' '.join(tokens)
            value_node = find_first(node, tag='span')
            metrics[label] = value_node.get_text().strip() if value_node else ''

        for label in ['Precision', 'Recall', 'F1 Score']:
            self.assertIn(label, metrics)
            self.assertRegex(metrics[label], r"^\d+\.\d+")

        tables = find_all(root, lambda n: n.tag == 'table')
        self.assertGreaterEqual(len(tables), 1)
        self.assertGreater(count_descendants(tables[-1], 'tr'), 1)

    def test_generate_interpretation_report_missing_feature(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            generate_interpretation_report(
                self.data[['feature1']],
                self.model,
                self.X_test[['feature1']],
                self.y_test,
            )

        self.assertIn('Feature names', buffer.getvalue())
        self.assertFalse(self.reports['interpretation'].exists())

    def test_suggest_further_analysis(self):
        suggest_further_analysis(self.data)
        self.assertTrue(self.reports['further'].exists())

        root = parse_html_tree(self.reports['further'])

        title = find_first(root, tag='h1')
        self.assertIsNotNone(title)
        self.assertEqual(title.get_text().strip(), 'Suggestions for Further Analysis')

        suggestion_items = find_all(root, lambda n: n.tag == 'li')
        self.assertGreaterEqual(len(suggestion_items), 5)

if __name__ == '__main__':
    unittest.main()
