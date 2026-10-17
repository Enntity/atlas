# SPDX-License-Identifier: AGPL-3.0-only
import unittest

from behavior_schema import schema_errors, validate_json_text

SCHEMA = {'type': 'object', 'additionalProperties': False,
          'required': ['city', 'population', 'tags', 'coastal'],
          'properties': {'city': {'type': 'string', 'enum': ['Tokyo', 'Paris']},
                         'population': {'type': 'integer', 'minimum': 1},
                         'tags': {'type': 'array', 'minItems': 1,
                                  'items': {'type': 'string', 'minLength': 1}},
                         'coastal': {'type': 'boolean'}}}


class SchemaTests(unittest.TestCase):
    def test_conforming_document_passes(self):
        value = validate_json_text(
            '{"city":"Tokyo","population":14000000,"tags":["capital"],"coastal":true}', SCHEMA)
        self.assertEqual(value['city'], 'Tokyo')

    def test_each_violation_is_reported(self):
        bad = {
            'missing': '{"city":"Tokyo","population":1,"tags":["a"]}',
            'extra': '{"city":"Tokyo","population":1,"tags":["a"],"coastal":true,"x":1}',
            'enum': '{"city":"Rome","population":1,"tags":["a"],"coastal":true}',
            'bool-as-int': '{"city":"Tokyo","population":true,"tags":["a"],"coastal":true}',
            'float-as-int': '{"city":"Tokyo","population":1.5,"tags":["a"],"coastal":true}',
            'minimum': '{"city":"Tokyo","population":0,"tags":["a"],"coastal":true}',
            'min-items': '{"city":"Tokyo","population":1,"tags":[],"coastal":true}',
            'item-type': '{"city":"Tokyo","population":1,"tags":[3],"coastal":true}',
            'not-object': '["Tokyo"]',
            'duplicate': '{"city":"Tokyo","city":"Paris","population":1,"tags":["a"],"coastal":true}',
            'nan': '{"city":"Tokyo","population":NaN,"tags":["a"],"coastal":true}',
            'fenced': '```json\n{"city":"Tokyo"}\n```',
        }
        for name, text in bad.items():
            with self.assertRaises(ValueError, msg=name):
                validate_json_text(text, SCHEMA)

    def test_unsupported_keywords_fail_closed(self):
        with self.assertRaises(ValueError):
            schema_errors({}, {'type': 'object', 'patternProperties': {}})
        with self.assertRaises(ValueError):
            schema_errors({}, {'type': 'object', 'additionalProperties': {'type': 'string'}})


if __name__ == '__main__':
    unittest.main()
