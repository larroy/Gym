# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib.metadata
import os
import select
import shlex
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from pytest import MonkeyPatch, raises

import nemo_gym.cli.setup_command
from nemo_gym.cli._venv_setup import SETUP_COMPLETE_MARKER, setup_environment
from nemo_gym.cli.setup_command import (
    _get_nemo_gym_install_flags,
    _get_nemo_gym_version_spec,
    get_venv_path,
    run_command,
    setup_env_command,
)
from nemo_gym.global_config import UV_VENV_DIR_KEY_NAME
from tests.unit_tests.test_global_config import TestGlobalConfig as _TestGlobalConfig


class TestCLISetupCommandSetupEnvCommand:
    def _installation_command(self, command: str) -> str:
        # Keep the dependency-command assertions independent of the shell quoting
        # used to pass that command as one argument to the setup runner.
        args = shlex.split(command)
        assert args[0] == "cd"
        assert args[2:4] == ["&&", sys.executable]
        assert Path(args[4]) == Path(nemo_gym.cli.setup_command.__file__).with_name("_venv_setup.py")
        venv = Path(args[args.index("--venv") + 1])
        assert args[-3:] == ["&&", "source", str(venv / "bin/activate")]
        return f"cd {args[1]} && {args[args.index('--command') + 1]}"

    def _setup_server_dir(self, tmp_path: Path) -> Path:
        server_dir = tmp_path / "first_level" / "second_level"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        (tmp_path / "pyproject.toml").write_text("")

        return server_dir.absolute()

    def _debug_global_config_dict(self, tmp_path: Path) -> dict:
        return _TestGlobalConfig._default_global_config_dict_values.fget(None) | {UV_VENV_DIR_KEY_NAME: str(tmp_path)}

    def test_sanity(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path),
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_requirements_uses_server_local_overrides(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "overrides.txt").write_text("dependency==2\n")

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path),
            prefix="my server name",
        )

        assert "uv pip install --override overrides.txt -r requirements.txt" in self._installation_command(
            actual_command
        )

    def test_reuse_decision_is_deferred_until_execution(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        config = self._debug_global_config_dict(tmp_path) | {"skip_venv_if_present": True}
        before = setup_env_command(server_dir, config, "policy")

        (server_dir / ".venv/bin").mkdir(parents=True)
        (server_dir / ".venv/bin/python").touch()
        (server_dir / ".venv/bin/activate").touch()
        (server_dir / ".venv" / SETUP_COMPLETE_MARKER).touch()

        assert setup_env_command(server_dir, config, "policy") == before
        assert "--skip-if-ready" in shlex.split(before)
        assert "uv pip install" in self._installation_command(before)

    def test_skips_install_still_installs_when_venv_missing(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        # No {server_dir}/.venv.
        # (server_dir / ".venv/bin").mkdir(parents=True)
        # (server_dir / ".venv/bin/python").write_text("")
        # (server_dir / ".venv/bin/activate").write_text("")

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"skip_venv_if_present": True},
            prefix="my server name",
        )

        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_head_server_deps(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"head_server_deps": ["dep 1", "dep 2"]},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -r requirements.txt dep 1 dep 2 > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_python_version(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"python_version": "my python version"},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'my python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_uv_pip_set_python(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"uv_pip_set_python": True},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --python {server_dir}/.venv/bin/python -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_pip_install_verbose(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"pip_install_verbose": True},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -v -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_pyproject_requirements_raises_error(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "pyproject.toml").write_text("")

        with raises(RuntimeError, match="Found both pyproject.toml and requirements.txt"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )

    def test_missing_pyproject_requirements_raises_error(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "requirements.txt").unlink()

        with raises(RuntimeError, match="Missing pyproject.toml or requirements.txt"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )

    def test_pyproject(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "pyproject.toml").write_text("")
        (server_dir / "requirements.txt").unlink()

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path),
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install '-e .' ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_uv_venv_dir_with_install(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        uv_venv_dir = tmp_path / "uv_venv_dir"

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"uv_venv_dir": str(uv_venv_dir)},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {uv_venv_dir}/first_level/second_level/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {uv_venv_dir}/first_level/second_level/.venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_uv_venv_dir_path_is_shared_with_cleanup(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        uv_venv_dir = tmp_path / "uv_venv_dir"

        actual_path = get_venv_path(
            server_dir,
            self._debug_global_config_dict(tmp_path) | {"uv_venv_dir": str(uv_venv_dir)},
        )

        assert actual_path == uv_venv_dir / "first_level/second_level/.venv"

    @pytest.mark.parametrize("version", ["0.3.0", "0.3.0rc0", "1.0.0", "2.1.3rc1"])
    def test_installs_from_pypi_when_not_editable(
        self, tmp_path: Path, version: str, monkeypatch: MonkeyPatch
    ) -> None:
        server_dir = (tmp_path / "first_level" / "second_level").absolute()
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        with patch("importlib.metadata.version", return_value=version):
            actual_command = setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && (echo 'nemo-gym=={version}' && grep -v -F '../..' requirements.txt) | uv pip install -r /dev/stdin ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    @pytest.mark.parametrize("version", ["0.3.0", "0.3.0rc0", "1.0.0", "2.1.3rc1"])
    def test_installs_from_pypi_when_not_editable_pyproject(
        self, tmp_path: Path, version: str, monkeypatch: MonkeyPatch
    ) -> None:
        server_dir = (tmp_path / "first_level" / "second_level").absolute()
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text("")
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        with patch("importlib.metadata.version", return_value=version):
            actual_command = setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install nemo-gym=={version} && uv pip install --no-sources '-e .' ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_custom_venv_path_is_passed_to_setup_runner(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        uv_venv_dir = tmp_path / "custom venvs"
        config = self._debug_global_config_dict(tmp_path) | {
            "skip_venv_if_present": True,
            "uv_venv_dir": str(uv_venv_dir),
        }
        args = shlex.split(setup_env_command(server_dir, config, "policy"))
        assert args[args.index("--venv") + 1] == str(uv_venv_dir / "first_level/second_level/.venv")
        assert "--skip-if-ready" in args


def _setup_fixture(root: Path) -> tuple[Path, dict, dict]:
    """Exercise the generated Bash command with a local, controllable uv substitute."""
    server_dir = root / "source with spaces" / "responses_api_models" / "model"
    server_dir.mkdir(parents=True)
    (server_dir.parent.parent / "pyproject.toml").touch()
    (server_dir / "requirements.txt").write_text("pytest\n")
    config = _TestGlobalConfig._default_global_config_dict_values.fget(None) | {
        "python_version": sys.executable,
        "uv_venv_dir": str(root / "venvs with spaces"),
        "skip_venv_if_present": True,
    }
    bin_dir = root / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "fake_uv.py"
    fake_uv.write_text(
        r"""import os
import shlex
import sys
from pathlib import Path

if sys.argv[1] == "venv":
    venv = Path(sys.argv[-1])
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    python = venv / "bin/python"
    if not python.exists():
        python.symlink_to(sys.executable)
    bin_path = shlex.quote(str(venv / "bin"))
    (venv / "bin/activate").write_text(
        f"export VIRTUAL_ENV={shlex.quote(str(venv))}\n"
        f'export PATH={bin_path}:"$PATH"\n'
    )
elif sys.argv[1:3] == ["pip", "install"]:
    venv = Path(os.environ["VIRTUAL_ENV"])
    with (venv / "install-attempts").open("a") as log:
        log.write("install\n")
    if "INSTALL_GATE" in os.environ:
        with open(os.environ["INSTALL_READY"], "wb", buffering=0) as ready:
            ready.write(b"1")
        with open(os.environ["INSTALL_GATE"], "rb", buffering=0) as gate:
            assert gate.read(1) == b"1"
    if "FAIL_INSTALL" in os.environ:
        sys.exit(int(os.environ["FAIL_INSTALL"]))
    (venv / "dependencies-installed").touch()
else:
    raise AssertionError(sys.argv)
"""
    )
    uv = bin_dir / "uv"
    uv.write_text(f'#!/bin/bash\nexec {shlex.join([sys.executable, str(fake_uv)])} "$@"\n')
    uv.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return server_dir, config, env


def _server_command(server_dir: Path, config: dict, name: str) -> str:
    server = (
        "import os; from pathlib import Path; "
        "assert (Path(os.environ['VIRTUAL_ENV']) / 'dependencies-installed').is_file(); "
        f"print({name!r}, flush=True); "
        "gate = os.environ.get('SERVER_GATE'); "
        "gate is None or open(gate, 'rb', buffering=0).read(1)"
    )
    return f"{setup_env_command(server_dir, config, name)} && python -c {shlex.quote(server)}"


@contextmanager
def _setup_process(command: str | list[str], server_dir: Path, env: dict):
    args = ["/bin/bash", "-c", command] if isinstance(command, str) else command
    process = subprocess.Popen(
        args,
        cwd=server_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        yield process
    finally:
        # Reap every child even when an assertion or a timeout interrupts a test.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=10)


@contextmanager
def _installation_gate(root: Path):
    ready = root / "install-ready"
    release = root / "install-release"
    os.mkfifo(ready)
    os.mkfifo(release)
    ready_fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    release_fd = os.open(release, os.O_RDWR | os.O_NONBLOCK)
    try:
        yield ready_fd, release_fd, {"INSTALL_READY": str(ready), "INSTALL_GATE": str(release)}
    finally:
        os.close(ready_fd)
        os.close(release_fd)


def _wait_for_install(ready_fd: int) -> None:
    assert select.select([ready_fd], [], [], 10)[0], "installer did not reach the barrier"
    assert os.read(ready_fd, 1) == b"1"


def _expect_output(process: subprocess.Popen, text: str) -> None:
    assert select.select([process.stdout], [], [], 10)[0], f"process did not report {text!r}"
    assert text in process.stdout.readline()


def _expect_success(process: subprocess.Popen, name: str) -> None:
    output, _ = process.communicate(timeout=10)
    assert process.returncode == 0, output
    assert name in output


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("generate_early", [False, True])
def test_shared_venv_waits_for_install_before_starting_servers(
    tmp_path: Path, alias: bool, generate_early: bool
) -> None:
    server_dir, config, env = _setup_fixture(tmp_path)
    venv = get_venv_path(server_dir, config)
    other_config = config.copy()
    if alias:
        root = Path(config["uv_venv_dir"])
        root.mkdir()
        link = tmp_path / "venv alias"
        link.symlink_to(root, target_is_directory=True)
        other_config["uv_venv_dir"] = str(link)
    # Cover both scheduling orders: commands built before setup starts, and
    # a judge command built while the policy's venv is only partially installed.
    policy_command = _server_command(server_dir, config, "POLICY_STARTED")
    judge_command = _server_command(server_dir, other_config, "JUDGE_STARTED") if generate_early else None
    with _installation_gate(tmp_path) as (ready, release, gate_env):
        with _setup_process(policy_command, server_dir, env | gate_env) as policy:
            _wait_for_install(ready)
            assert (venv / "bin/python").is_file()
            assert (venv / "bin/activate").is_file()
            assert not (venv / SETUP_COMPLETE_MARKER).exists()
            if judge_command is None:
                judge_command = _server_command(server_dir, other_config, "JUDGE_STARTED")
            with _setup_process(judge_command, server_dir, env) as judge:
                _expect_output(judge, "Waiting for virtual environment setup:")
                assert not select.select([policy.stdout, judge.stdout], [], [], 0)[0]
                os.write(release, b"1")
                _expect_success(policy, "POLICY_STARTED")
                _expect_success(judge, "JUDGE_STARTED")
    assert (venv / "install-attempts").read_text() == "install\n"
    assert (venv / SETUP_COMPLETE_MARKER).is_file()


def test_different_venvs_install_independently(tmp_path: Path) -> None:
    first_dir, first_config, first_env = _setup_fixture(tmp_path / "first")
    other_dir, other_config, other_env = _setup_fixture(tmp_path / "other")
    with _installation_gate(tmp_path) as (ready, release, gate_env):
        with _setup_process(
            _server_command(first_dir, first_config, "FIRST"), first_dir, first_env | gate_env
        ) as first:
            _wait_for_install(ready)
            with _setup_process(_server_command(other_dir, other_config, "OTHER"), other_dir, other_env) as other:
                _expect_success(other, "OTHER")
            assert first.poll() is None
            os.write(release, b"1")
            _expect_success(first, "FIRST")


@pytest.mark.parametrize("state", ["legacy", "ready", "missing_python", "missing_activate"])
def test_existing_venv_is_reused_only_after_complete_setup(tmp_path: Path, state: str) -> None:
    server_dir, config, env = _setup_fixture(tmp_path)
    venv = get_venv_path(server_dir, config)
    with _setup_process(_server_command(server_dir, config, "INITIAL"), server_dir, env) as initial:
        _expect_success(initial, "INITIAL")
    (venv / "keep-me").write_text("user data")
    if state == "legacy":
        (venv / SETUP_COMPLETE_MARKER).unlink()
    elif state == "missing_python":
        (venv / "bin/python").unlink()
    elif state == "missing_activate":
        (venv / "bin/activate").unlink()
    with _setup_process(_server_command(server_dir, config, "REUSED"), server_dir, env) as reused:
        _expect_success(reused, "REUSED")
    assert (venv / "install-attempts").read_text() == "install\n" * (1 if state == "ready" else 2)
    assert (venv / "keep-me").read_text() == "user data"
    assert (venv / SETUP_COMPLETE_MARKER).is_file()


@pytest.mark.parametrize("already_ready", [False, True])
def test_failed_install_stops_server_and_can_be_retried(tmp_path: Path, already_ready: bool) -> None:
    server_dir, config, env = _setup_fixture(tmp_path)
    venv = get_venv_path(server_dir, config)
    if already_ready:
        with _setup_process(_server_command(server_dir, config, "INITIAL"), server_dir, env) as initial:
            _expect_success(initial, "INITIAL")
    # A forced reinstall must invalidate an old completion marker before work starts.
    command = _server_command(server_dir, config | {"skip_venv_if_present": False}, "MUST_NOT_START")
    with _setup_process(command, server_dir, env | {"FAIL_INSTALL": "7"}) as failed:
        output, _ = failed.communicate(timeout=10)
        assert failed.returncode == 7, output
        assert "MUST_NOT_START" not in output
    assert not (venv / SETUP_COMPLETE_MARKER).exists()
    with _setup_process(_server_command(server_dir, config, "RETRIED"), server_dir, env) as retried:
        _expect_success(retried, "RETRIED")
    assert (venv / SETUP_COMPLETE_MARKER).is_file()
    assert (venv / "install-attempts").read_text() == "install\n" * (3 if already_ready else 2)


@pytest.mark.parametrize("signal_number", [signal.SIGTERM, signal.SIGKILL])
def test_surviving_installer_keeps_lock_when_wrapper_dies(tmp_path: Path, signal_number: int) -> None:
    server_dir, config, env = _setup_fixture(tmp_path)
    venv = get_venv_path(server_dir, config)
    # Start the actual runner directly so the signal targets it, not an outer shell.
    wrapper = shlex.split(setup_env_command(server_dir, config, "policy"))[3:-3]
    with _installation_gate(tmp_path) as (ready, release, gate_env):
        with _setup_process(wrapper, server_dir, env | gate_env) as killed:
            _wait_for_install(ready)
            killed.send_signal(signal_number)
            assert killed.wait(timeout=10) == -signal_number
            assert not (venv / SETUP_COMPLETE_MARKER).exists()
            with _setup_process(_server_command(server_dir, config, "RETRIED"), server_dir, env) as retried:
                _expect_output(retried, "Waiting for virtual environment setup:")
                os.write(release, b"1")
                _expect_success(retried, "RETRIED")
    assert (venv / "install-attempts").read_text() == "install\ninstall\n"
    assert (venv / SETUP_COMPLETE_MARKER).is_file()


def test_running_server_does_not_hold_setup_lock(tmp_path: Path) -> None:
    server_dir, config, env = _setup_fixture(tmp_path)
    gate = tmp_path / "server-gate"
    os.mkfifo(gate)
    gate_fd = os.open(gate, os.O_RDWR | os.O_NONBLOCK)
    try:
        with _setup_process(
            _server_command(server_dir, config, "POLICY_STARTED"), server_dir, env | {"SERVER_GATE": str(gate)}
        ) as policy:
            _expect_output(policy, "POLICY_STARTED")
            with _setup_process(_server_command(server_dir, config, "JUDGE_STARTED"), server_dir, env) as judge:
                _expect_success(judge, "JUDGE_STARTED")
            assert policy.poll() is None
            os.write(gate_fd, b"1")
            output, _ = policy.communicate(timeout=10)
            assert policy.returncode == 0, output
    finally:
        os.close(gate_fd)


def test_successful_command_without_a_complete_venv_is_not_trusted(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    with raises(RuntimeError, match="did not create"):
        setup_environment(venv, "true", skip_if_ready=True)
    assert not (venv / SETUP_COMPLETE_MARKER).exists()


def test_signalled_installer_returns_failure_without_a_marker(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    assert setup_environment(venv, "kill -TERM $$", skip_if_ready=True) == 128 + signal.SIGTERM
    assert not (venv / SETUP_COMPLETE_MARKER).exists()


class TestCLISetupCommandRunCommand:
    def _setup(self, monkeypatch: MonkeyPatch) -> tuple[MagicMock, MagicMock]:
        Popen_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.cli.setup_command, "Popen", Popen_mock)

        get_global_config_dict_mock = MagicMock(return_value={"uv_cache_dir": "default uv cache dir"})
        monkeypatch.setattr(nemo_gym.cli.setup_command, "get_global_config_dict", get_global_config_dict_mock)

        monkeypatch.setattr(nemo_gym.cli.setup_command, "environ", dict())

        monkeypatch.setattr(nemo_gym.cli.setup_command, "stdout", "stdout")
        monkeypatch.setattr(nemo_gym.cli.setup_command, "stderr", "stderr")

        return Popen_mock, get_global_config_dict_mock

    def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            # Default (no project_root): only the server dir is on PYTHONPATH.
            env={"PYTHONPATH": "/my path", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_custom_pythonpath(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)
        monkeypatch.setattr(nemo_gym.cli.setup_command, "environ", {"PYTHONPATH": "existing pythonpath"})

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path:existing pythonpath", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_project_root_added_to_pythonpath(self, monkeypatch: MonkeyPatch) -> None:
        # Opt-in: callers that need `resources_servers.<name>`-style imports (e.g. gym env test) pass
        # the project root, which is appended after the server dir.
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
            project_root=Path("/root"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server:/root", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_custom_uv_cache_dir(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {"uv_cache_dir": "my uv cache dir"}

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path", "UV_CACHE_DIR": "my uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_supplied_config_and_streams_avoid_global_config(self, monkeypatch: MonkeyPatch) -> None:
        popen, get_global_config = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
            global_config_dict={"uv_cache_dir": "isolated cache"},
            stdout_target="isolated stdout",
            stderr_target="isolated stderr",
        )

        get_global_config.assert_not_called()
        assert popen.call_args.kwargs["env"]["UV_CACHE_DIR"] == "isolated cache"
        assert popen.call_args.kwargs["stdout"] == "isolated stdout"
        assert popen.call_args.kwargs["stderr"] == "isolated stderr"


class TestGetNemoGymInstallFlags:
    """Test _get_nemo_gym_install_flags helper function."""

    def test_no_env_vars_returns_empty(self, monkeypatch: MonkeyPatch) -> None:
        """When no env vars are set, should return empty string."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == ""

    def test_prerelease_flag(self, monkeypatch: MonkeyPatch) -> None:
        """When NEMO_GYM_ALLOW_PRERELEASE=true, should add --pre and --index-strategy."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--pre --index-strategy unsafe-best-match 'fastapi<1.0' "

    def test_prerelease_false(self, monkeypatch: MonkeyPatch) -> None:
        """When NEMO_GYM_ALLOW_PRERELEASE=false, should not add flags."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "false")
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == ""

    def test_index_url(self, monkeypatch: MonkeyPatch) -> None:
        """Should include UV_INDEX_URL if set."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.setenv("UV_INDEX_URL", "https://test.pypi.org/simple/")
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--index-url https://test.pypi.org/simple/ "

    def test_extra_index_url(self, monkeypatch: MonkeyPatch) -> None:
        """Should include UV_EXTRA_INDEX_URL if set."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://pypi.org/simple/")
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--extra-index-url https://pypi.org/simple/ "

    def test_explicit_index_strategy(self, monkeypatch: MonkeyPatch) -> None:
        """Explicit UV_INDEX_STRATEGY should override auto-set from prerelease."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.setenv("UV_INDEX_STRATEGY", "first-match")

        flags = _get_nemo_gym_install_flags()
        # Should have --pre but use explicit strategy, not auto-set unsafe-best-match
        assert flags == "--pre 'fastapi<1.0' --index-strategy first-match "

    def test_all_flags_combined(self, monkeypatch: MonkeyPatch) -> None:
        """Test all flags together."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.setenv("UV_INDEX_URL", "https://test.pypi.org/simple/")
        monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://pypi.org/simple/")
        monkeypatch.setenv("UV_INDEX_STRATEGY", "unsafe-best-match")

        flags = _get_nemo_gym_install_flags()
        assert (
            flags
            == "--pre 'fastapi<1.0' --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match "
        )


class TestGetNemoGymVersionSpec:
    """Test _get_nemo_gym_version_spec helper function."""

    def test_editable_install_returns_empty(self) -> None:
        """For editable installs, should return empty string (no version pinning)."""
        version_spec = _get_nemo_gym_version_spec(is_editable_install=True)
        assert version_spec == ""

    def test_non_editable_detects_version(self) -> None:
        """For non-editable installs, should detect and pin to parent version."""
        with patch("importlib.metadata.version", return_value="0.2.1rc0"):
            version_spec = _get_nemo_gym_version_spec(is_editable_install=False)
            assert version_spec == "==0.2.1rc0"

    def test_non_editable_stable_version(self) -> None:
        """Should work with stable versions too."""
        with patch("importlib.metadata.version", return_value="0.2.0"):
            version_spec = _get_nemo_gym_version_spec(is_editable_install=False)
            assert version_spec == "==0.2.0"

    def test_package_not_found_returns_empty(self) -> None:
        """If nemo-gym is not installed, should return empty string gracefully."""
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            version_spec = _get_nemo_gym_version_spec(is_editable_install=False)
            assert version_spec == ""


class TestCLISetupCommandRunCommandTeeLog(TestCLISetupCommandRunCommand):
    def test_tee_logs_with_server_name(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {
            "uv_cache_dir": "default uv cache dir",
            "nemo_gym_log_dir": "/tmp/gym_logs",
        }

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
            server_name="my_resources/my_server",
        )

        expected_args = call(
            "set -o pipefail; (my command) 2>&1 | tee -a /tmp/gym_logs/my_resources_my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_tee_logs_falls_back_to_dir_name(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {
            "uv_cache_dir": "default uv cache dir",
            "nemo_gym_log_dir": "/tmp/gym_logs",
        }

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
        )

        expected_args = call(
            "set -o pipefail; (my command) 2>&1 | tee -a /tmp/gym_logs/my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args
