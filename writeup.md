Answers to written responses. 

## P1: Understanding Unicode
(a) What Unicode character does `chr(0)` return?
**Ans:** `\x00`, the null character where all bits in the byte are set to 0. Its used to denote end of strings in C, for example.

(b) How does this character’s string representation (`__repr__()`) differ from its printed representa-
tion?
**Ans:** the repr is escaped so as to actually print something where as the str is not:
```python
>>> chr(0).__repr__(), chr(0).__str__()
("'\\x00'", '\x00')
```

(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
**Ans:** It doesn't appear because its not considered a printable character. 


## P2: Unicode Encodings

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.

**Ans:** UTF-8 is more efficient with space, taking 1 byte for ASCII characters, and up to 4 for other chars. UTF-16 uses 2-4 bytes, while UTF-32 a fixed 4 bytes.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello
```
**Ans:** The function assumes that each byte in the bytestring represents a UTF-8 character, when we know that there are characters that take up to 4. For example, 
```python
>>> decode_utf8_bytes_to_str_wrong("helló".encode("utf-8"))
```
breaks because 'ó' is represented by two bytes.

(c) Give a two byte sequence that does not decode to any Unicode character(s).
Deliverable: An example, with a one-sentence explanation.

**Ans:** `bytes([0, 128]).decode('utf-8')` because bytes 128-256 are reserved for multi-byte characters, so they must be leading. By themselves they are invalid.


