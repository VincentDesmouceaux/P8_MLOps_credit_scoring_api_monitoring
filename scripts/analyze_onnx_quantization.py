import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "credit_scoring_model.onnx"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = (
    REPORTS_DIR
    / "onnx_quantization_analysis.json"
)


# -------------------------------------------------------------------
# Opérateurs généralement ciblés par la quantification ONNX Runtime
# -------------------------------------------------------------------

COMMON_QUANTIZABLE_OPERATORS = {
    "MatMul",
    "Gemm",
    "Conv",
    "Attention",
    "LSTM",
    "GRU",
}

TREE_OPERATORS = {
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
}


# -------------------------------------------------------------------
# Chargement du modèle
# -------------------------------------------------------------------

def load_onnx_model() -> onnx.ModelProto:
    """
    Charge et valide le modèle ONNX.
    """
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modèle ONNX introuvable : "
            f"{ONNX_MODEL_PATH}"
        )

    print("Chargement du modèle ONNX...")

    try:
        model = onnx.load(
            str(ONNX_MODEL_PATH)
        )

        onnx.checker.check_model(
            model
        )

    except Exception as error:
        raise RuntimeError(
            "Impossible de charger ou valider "
            "le modèle ONNX."
        ) from error

    print("Modèle ONNX valide.")

    return model


# -------------------------------------------------------------------
# Analyse du graphe
# -------------------------------------------------------------------

def analyze_operators(
    model: onnx.ModelProto,
) -> dict[str, Any]:
    """
    Analyse les opérateurs présents dans le graphe ONNX.

    L'objectif est de déterminer si le modèle contient
    des opérateurs typiquement ciblés par les mécanismes
    de quantification ONNX Runtime.
    """
    operator_counts = Counter(
        node.op_type
        for node in model.graph.node
    )

    operators = sorted(
        operator_counts
    )

    quantizable_operators = sorted(
        operator
        for operator in operators
        if operator
        in COMMON_QUANTIZABLE_OPERATORS
    )

    tree_operators = sorted(
        operator
        for operator in operators
        if operator
        in TREE_OPERATORS
    )

    return {
        "node_count": len(
            model.graph.node
        ),
        "operator_counts": dict(
            sorted(
                operator_counts.items()
            )
        ),
        "operators": operators,
        "common_quantizable_operators": (
            quantizable_operators
        ),
        "tree_operators": (
            tree_operators
        ),
    }


# -------------------------------------------------------------------
# Analyse des types des initializers
# -------------------------------------------------------------------

def analyze_initializers(
    model: onnx.ModelProto,
) -> dict[str, Any]:
    """
    Analyse les tenseurs constants présents dans le graphe.

    Cette information permet notamment de déterminer
    si le modèle contient des poids sous forme de tenseurs
    susceptibles d'être concernés par une quantification
    classique.
    """
    initializer_count = len(
        model.graph.initializer
    )

    data_types = Counter(
        onnx.TensorProto.DataType.Name(
            initializer.data_type
        )
        for initializer
        in model.graph.initializer
    )

    return {
        "initializer_count": (
            initializer_count
        ),
        "initializer_data_types": dict(
            sorted(
                data_types.items()
            )
        ),
    }


# -------------------------------------------------------------------
# Taille du modèle
# -------------------------------------------------------------------

def get_model_size() -> dict[str, float]:
    """
    Retourne la taille de l'artefact ONNX.
    """
    size_bytes = (
        ONNX_MODEL_PATH.stat().st_size
    )

    return {
        "size_bytes": size_bytes,
        "size_kb": round(
            size_bytes / 1024,
            3,
        ),
        "size_mb": round(
            size_bytes
            / (1024 * 1024),
            3,
        ),
    }


# -------------------------------------------------------------------
# Diagnostic
# -------------------------------------------------------------------

def build_quantization_diagnostic(
    operator_analysis: dict[str, Any],
    initializer_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Produit un diagnostic prudent sur la pertinence
    d'une quantification classique ONNX Runtime.

    Cette fonction ne prétend pas qu'une transformation
    est impossible. Elle indique si le graphe ressemble
    à un candidat classique pour les outils de
    quantification ONNX Runtime.
    """
    quantizable_operators = (
        operator_analysis[
            "common_quantizable_operators"
        ]
    )

    tree_operators = (
        operator_analysis[
            "tree_operators"
        ]
    )

    initializer_count = (
        initializer_analysis[
            "initializer_count"
        ]
    )

    is_tree_model = bool(
        tree_operators
    )

    has_common_quantizable_ops = bool(
        quantizable_operators
    )

    has_initializers = (
        initializer_count > 0
    )

    if (
        is_tree_model
        and not has_common_quantizable_ops
    ):
        recommendation = (
            "La quantification INT8 classique d'ONNX Runtime "
            "n'est pas une stratégie prioritaire pour ce graphe. "
            "Le modèle repose sur un opérateur d'ensemble d'arbres "
            "et ne contient pas d'opérateurs matriciels "
            "classiquement ciblés par la quantification."
        )

        candidate_for_standard_quantization = False

    elif has_common_quantizable_ops:
        recommendation = (
            "Le graphe contient des opérateurs couramment ciblés "
            "par la quantification ONNX Runtime. "
            "Une expérimentation de quantification peut être "
            "pertinente, sous réserve d'une validation des "
            "performances et de la précision."
        )

        candidate_for_standard_quantization = True

    else:
        recommendation = (
            "Aucun opérateur couramment ciblé par la "
            "quantification ONNX Runtime n'a été identifié. "
            "Une analyse spécifique du graphe est recommandée "
            "avant toute transformation."
        )

        candidate_for_standard_quantization = False

    return {
        "tree_based_model_detected": (
            is_tree_model
        ),
        "common_quantizable_operators_detected": (
            has_common_quantizable_ops
        ),
        "initializers_detected": (
            has_initializers
        ),
        "candidate_for_standard_quantization": (
            candidate_for_standard_quantization
        ),
        "recommendation": recommendation,
    }


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main() -> None:
    print(
        "\n=== ANALYSE QUANTIFICATION ONNX ===\n"
    )

    model = load_onnx_model()

    operator_analysis = (
        analyze_operators(
            model
        )
    )

    initializer_analysis = (
        analyze_initializers(
            model
        )
    )

    model_size = get_model_size()

    diagnostic = (
        build_quantization_diagnostic(
            operator_analysis,
            initializer_analysis,
        )
    )

    # ---------------------------------------------------------------
    # Rapport
    # ---------------------------------------------------------------

    report = {
        "model_path": str(
            ONNX_MODEL_PATH
        ),
        "model_size": model_size,
        "graph": operator_analysis,
        "initializers": (
            initializer_analysis
        ),
        "quantization_diagnostic": (
            diagnostic
        ),
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------------
    # Affichage
    # ---------------------------------------------------------------

    print(
        "\n=== MODELE ===\n"
    )

    print(
        "Taille :",
        f"{model_size['size_mb']:.3f} MB",
    )

    print(
        "Nombre de noeuds :",
        operator_analysis[
            "node_count"
        ],
    )

    print(
        "\n=== OPERATEURS ===\n"
    )

    for operator, count in (
        operator_analysis[
            "operator_counts"
        ].items()
    ):
        print(
            f"- {operator}: {count}"
        )

    print(
        "\nOpérateurs d'arbres :",
        operator_analysis[
            "tree_operators"
        ],
    )

    print(
        "Opérateurs classiquement quantifiables :",
        operator_analysis[
            "common_quantizable_operators"
        ],
    )

    print(
        "\n=== INITIALIZERS ===\n"
    )

    print(
        "Nombre :",
        initializer_analysis[
            "initializer_count"
        ],
    )

    print(
        "Types :",
        initializer_analysis[
            "initializer_data_types"
        ],
    )

    print(
        "\n=== DIAGNOSTIC QUANTIFICATION ===\n"
    )

    print(
        "Modèle d'arbres détecté :",
        diagnostic[
            "tree_based_model_detected"
        ],
    )

    print(
        "Candidat quantification standard :",
        diagnostic[
            "candidate_for_standard_quantization"
        ],
    )

    print(
        "\nConclusion :"
    )

    print(
        diagnostic[
            "recommendation"
        ]
    )

    print(
        "\nRapport :",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()