#!/usr/bin/env python3
"""
Curate ML engineering dataset for PEFT fine-tuning.
Collects traces from Kaggle notebooks, common ML patterns, and MLE-bench training splits.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class MLETrainingExample:
    """Single training example for ML engineering."""
    instruction: str
    response: str
    tools_used: List[str]
    code: str
    metadata: Dict[str, Any]

def extract_kaggle_patterns(kaggle_notebook_path: Path) -> List[MLETrainingExample]:
    """Extract ML engineering patterns from Kaggle notebook."""
    examples = []
    
    if not kaggle_notebook_path.exists():
        return examples
    
    content = kaggle_notebook_path.read_text()
    
    # Extract code cells (simplified - actual notebooks are JSON)
    # This is a placeholder - real implementation would parse .ipynb JSON
    code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
    
    for code in code_blocks:
        # Identify common ML patterns
        if "import pandas" in code or "import numpy" in code:
            examples.append(MLETrainingExample(
                instruction="Load and explore a dataset",
                response=f"Here's how to load and explore data:\n\n```python\n{code}\n```",
                tools_used=["python_exec"],
                code=code,
                metadata={"source": "kaggle", "pattern": "data_loading"},
            ))
        
        if "from sklearn" in code or "import xgboost" in code or "import lightgbm" in code:
            examples.append(MLETrainingExample(
                instruction="Train a machine learning model",
                response=f"Here's a model training pipeline:\n\n```python\n{code}\n```",
                tools_used=["python_exec"],
                code=code,
                metadata={"source": "kaggle", "pattern": "model_training"},
            ))
        
        if ".to_csv" in code or "submission" in code.lower():
            examples.append(MLETrainingExample(
                instruction="Generate submission file",
                response=f"Here's how to create a submission:\n\n```python\n{code}\n```",
                tools_used=["python_exec", "fs_write"],
                code=code,
                metadata={"source": "kaggle", "pattern": "submission"},
            ))
    
    return examples

def create_common_patterns() -> List[MLETrainingExample]:
    """Create examples for common ML engineering patterns."""
    examples = []
    
    # Data loading pattern
    examples.append(MLETrainingExample(
        instruction="Load a CSV dataset and display basic information",
        response="""I'll load the dataset and show basic statistics:

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('train.csv')

# Display basic info
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
```""",
        tools_used=["python_exec", "fs_read"],
        code="""import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())""",
        metadata={"source": "synthetic", "pattern": "data_loading"},
    ))
    
    # Feature engineering pattern
    examples.append(MLETrainingExample(
        instruction="Create features from existing columns",
        response="""I'll create some common features:

```python
# Handle missing values
df['feature'].fillna(df['feature'].median(), inplace=True)

# Create interaction features
df['interaction'] = df['feature1'] * df['feature2']

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['categorical'])
```""",
        tools_used=["python_exec"],
        code="""df['feature'].fillna(df['feature'].median(), inplace=True)
df['interaction'] = df['feature1'] * df['feature2']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['categorical'])""",
        metadata={"source": "synthetic", "pattern": "feature_engineering"},
    ))
    
    # Model training pattern
    examples.append(MLETrainingExample(
        instruction="Train a scikit-learn model with cross-validation",
        response="""I'll train a model with cross-validation:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})')

# Fit on full training data
model.fit(X, y)
```""",
        tools_used=["python_exec"],
        code="""from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})')
model.fit(X, y)""",
        metadata={"source": "synthetic", "pattern": "model_training"},
    ))
    
    # Submission generation pattern
    examples.append(MLETrainingExample(
        instruction="Generate submission CSV in correct format",
        response="""I'll create the submission file:

```python
# Make predictions
predictions = model.predict(test_df)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print('Submission saved!')
```""",
        tools_used=["python_exec", "fs_write"],
        code="""predictions = model.predict(test_df)
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': predictions
})
submission.to_csv('submission.csv', index=False)
print('Submission saved!')""",
        metadata={"source": "synthetic", "pattern": "submission"},
    ))
    
    # Error handling pattern
    examples.append(MLETrainingExample(
        instruction="Fix a common Python error: KeyError when accessing DataFrame column",
        response="""I'll fix the KeyError by checking if the column exists:

```python
# Check available columns
print(df.columns.tolist())

# Use correct column name or handle missing column
if 'target' in df.columns:
    y = df['target']
else:
    print('Target column not found. Available columns:', df.columns.tolist())
```""",
        tools_used=["python_exec"],
        code="""print(df.columns.tolist())
if 'target' in df.columns:
    y = df['target']
else:
    print('Target column not found. Available columns:', df.columns.tolist())""",
        metadata={"source": "synthetic", "pattern": "error_handling"},
    ))
    
    return examples

def format_for_training(examples: List[MLETrainingExample]) -> List[Dict[str, Any]]:
    """Format examples for training (e.g., Alpaca format)."""
    formatted = []
    
    for ex in examples:
        # Format as instruction-following example
        formatted.append({
            "instruction": ex.instruction,
            "input": "",
            "output": ex.response,
            "tools": ex.tools_used,
            "code": ex.code,
            "metadata": ex.metadata,
        })
    
    return formatted

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Curate ML engineering dataset")
    parser.add_argument("--kaggle-notebooks", type=str, help="Directory with Kaggle notebooks")
    parser.add_argument("--output", type=str, default="./finetune_data.jsonl", help="Output file")
    parser.add_argument("--include-synthetic", action="store_true", default=True, help="Include synthetic patterns")
    
    args = parser.parse_args()
    
    examples = []
    
    # Add synthetic/common patterns
    if args.include_synthetic:
        examples.extend(create_common_patterns())
    
    # Extract from Kaggle notebooks if provided
    if args.kaggle_notebooks:
        kaggle_dir = Path(args.kaggle_notebooks)
        for notebook_path in kaggle_dir.glob("*.ipynb"):
            examples.extend(extract_kaggle_patterns(notebook_path))
    
    # Format for training
    formatted = format_for_training(examples)
    
    # Save as JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Created {len(formatted)} training examples")
    print(f"✓ Saved to {output_path}")

if __name__ == "__main__":
    main()

