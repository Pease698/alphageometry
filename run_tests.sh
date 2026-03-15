# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

DATA=ag_ckpt_vocab
MELIAD_PATH=meliad_lib/meliad
export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH:origin:reuse

uv run python reuse/problem_test.py
uv run python reuse/geometry_test.py
uv run python reuse/graph_utils_test.py
uv run python reuse/numericals_test.py
uv run python reuse/graph_test.py
uv run python reuse/dd_test.py
uv run python reuse/ar_test.py
uv run python reuse/ddar_test.py
uv run python reuse/trace_back_test.py
uv run python alphageometry_test.py
uv run python origin/lm_inference_test.py --meliad_path=$MELIAD_PATH --data_path=$DATA
