"""Load functions straight out of the notebook so tests exercise the published source."""
import ast
import json
import pathlib

NOTEBOOK = pathlib.Path(__file__).resolve().parents[1] / "rag-biomedical-qa.ipynb"


def notebook_source(path=NOTEBOOK) -> str:
    """Concatenate the notebook's code cells, dropping shell and magic lines."""
    nb = json.loads(path.read_text())
    lines = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        skipping = False
        for line in "".join(cell["source"]).split("\n"):
            if skipping:
                skipping = line.rstrip().endswith("\\")
                continue
            if line.lstrip().startswith(("!", "%")):
                skipping = line.rstrip().endswith("\\")
                continue
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def load(names, env=None, path=NOTEBOOK):
    """Execute only the named top-level defs and assignments, over the given globals."""
    tree = ast.parse(notebook_source(path))
    wanted = set(names)
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            picked.append(node)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
                picked.append(node)
    missing = wanted - {
        n.name if isinstance(n, ast.FunctionDef)
        else next(t.id for t in n.targets if isinstance(t, ast.Name))
        for n in picked
    }
    if missing:
        raise AssertionError(f"not found in notebook: {sorted(missing)}")
    ns = dict(env or {})
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), ns)
    return ns


def calls_named(func_name, path=NOTEBOOK):
    """Every call to `func_name` in the notebook, as ast.Call nodes."""
    tree = ast.parse(notebook_source(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name == func_name:
                out.append(node)
    return out


def kwargs_of(call):
    """Keyword arguments of a call, as {name: source-text}."""
    return {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
