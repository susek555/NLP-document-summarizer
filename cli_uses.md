.venv/bin/python main.py --name "grok_iterative/attention" download 1706.03762
.venv/bin/python main.py --name "grok_iterative/attention" --llm "LLAMA_GROK" clean
.venv/bin/python main.py --name "grok_iterative/attention" --llm "LLAMA_GROK" summarize --method "ITERATIVE"
.venv/bin/python main.py --name "grok_iterative/attention" --llm "LLAMA_GROK" keywords --abs_type "ITERATIVE"
.venv/bin/python main.py --name "grok_iterative/attention" metrics --abs_type "ITERATIVE"

.venv/bin/python main.py --name "gemini31_map_reduce/attention" download 1706.03762
.venv/bin/python main.py --name "gemini31_map_reduce/attention" --llm "GEMINI_31_FLASH_LITE" clean
.venv/bin/python main.py --name "gemini31_map_reduce/attention" --llm "GEMINI_31_FLASH_LITE" summarize --method "MAP_REDUCE"
.venv/bin/python main.py --name "gemini31_map_reduce/attention" --llm "GEMINI_31_FLASH_LITE" keywords --abs_type "MAP_REDUCE"
.venv/bin/python main.py --name "gemini31_map_reduce/attention" metrics --abs_type "MAP_REDUCE"

.venv/bin/python main.py --name "gemini31_iterative/attention" download 1706.03762
.venv/bin/python main.py --name "gemini31_iterative/attention" --llm "GEMINI_31_FLASH_LITE" clean
.venv/bin/python main.py --name "gemini31_iterative/attention" --llm "GEMINI_31_FLASH_LITE" summarize --method "ITERATIVE"
.venv/bin/python main.py --name "gemini31_iterative/attention" --llm "GEMINI_31_FLASH_LITE" keywords --abs_type "ITERATIVE"
.venv/bin/python main.py --name "gemini31_iterative/attention" metrics --abs_type "ITERATIVE"

.venv/bin/python main.py --name "grok_map_reduce/attention" download 1706.03762
.venv/bin/python main.py --name "grok_map_reduce/attention" --llm "LLAMA_GROK" clean
.venv/bin/python main.py --name "grok_map_reduce/attention" --llm "LLAMA_GROK" summarize --method "MAP_REDUCE"
.venv/bin/python main.py --name "grok_map_reduce/attention" --llm "LLAMA_GROK" keywords --abs_type "MAP_REDUCE"
.venv/bin/python main.py --name "grok_map_reduce/attention" metrics --abs_type "MAP_REDUCE"