# SPDX-License-Identifier: AGPL-3.0-only
"""Minimal stdlib JSON-Schema subset checker for behavior gates.

Covers exactly the keywords the behavior suite declares in its tool and
response_format schemas: type (incl. lists), properties, required,
additionalProperties (bool), items, enum, const, minimum, maximum,
minLength, maxLength, minItems, maxItems. Any other keyword is rejected so
a schema can never silently pass unchecked. JSON booleans are never
integers/numbers here (Python's bool-is-int is excluded explicitly).
"""
import json

SUPPORTED = {'type', 'properties', 'required', 'additionalProperties', 'items',
             'enum', 'const', 'minimum', 'maximum', 'minLength', 'maxLength',
             'minItems', 'maxItems', 'description', 'title'}


def _is_type(value, name):
    if name == 'object':
        return isinstance(value, dict)
    if name == 'array':
        return isinstance(value, list)
    if name == 'string':
        return isinstance(value, str)
    if name == 'boolean':
        return isinstance(value, bool)
    if name == 'null':
        return value is None
    if name == 'integer':
        return type(value) is int
    if name == 'number':
        return type(value) in (int, float)
    raise ValueError('unsupported schema type: ' + repr(name))


def schema_errors(value, schema, path='$'):
    """Return a list of violations (empty when `value` conforms)."""
    if not isinstance(schema, dict):
        raise ValueError('schema must be an object at ' + path)
    unknown = set(schema) - SUPPORTED
    if unknown:
        raise ValueError('unsupported schema keywords at ' + path + ': ' + repr(sorted(unknown)))
    errors = []
    if 'type' in schema:
        names = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        if not any(_is_type(value, name) for name in names):
            return [path + ': expected ' + '/'.join(names) + ', got ' + type(value).__name__]
    if 'const' in schema and value != schema['const']:
        errors.append(path + ': const mismatch')
    if 'enum' in schema and not any(value == item and type(value) is type(item)
                                    for item in schema['enum']):
        errors.append(path + ': not in enum ' + repr(schema['enum']))
    if type(value) in (int, float):
        if 'minimum' in schema and value < schema['minimum']:
            errors.append(path + ': below minimum')
        if 'maximum' in schema and value > schema['maximum']:
            errors.append(path + ': above maximum')
    if isinstance(value, str):
        if 'minLength' in schema and len(value) < schema['minLength']:
            errors.append(path + ': shorter than minLength')
        if 'maxLength' in schema and len(value) > schema['maxLength']:
            errors.append(path + ': longer than maxLength')
    if isinstance(value, list):
        if 'minItems' in schema and len(value) < schema['minItems']:
            errors.append(path + ': fewer than minItems')
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            errors.append(path + ': more than maxItems')
        if 'items' in schema:
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, schema['items'], path + '[' + str(index) + ']'))
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                errors.append(path + ': missing required ' + repr(key))
        extra = schema.get('additionalProperties', True)
        if extra is not True and extra is not False:
            raise ValueError('additionalProperties must be boolean at ' + path)
        for key, item in value.items():
            if key in properties:
                errors.extend(schema_errors(item, properties[key], path + '.' + key))
            elif extra is False:
                errors.append(path + ': unexpected property ' + repr(key))
    return errors


def parse_strict_json(text):
    """Parse JSON text rejecting duplicate keys and non-finite numbers."""
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError('duplicate JSON keys')
        return result

    def reject(constant):
        raise ValueError('non-finite JSON constant ' + constant)

    if not isinstance(text, str):
        raise ValueError('JSON text required')
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def validate_json_text(text, schema):
    """Parse `text` strictly and require full conformance to `schema`."""
    value = parse_strict_json(text)
    errors = schema_errors(value, schema)
    if errors:
        raise ValueError('schema violation: ' + '; '.join(errors))
    return value
