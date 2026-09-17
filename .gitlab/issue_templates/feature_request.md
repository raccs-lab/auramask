---
name: Feature Request
about: Propose a new feature, API addition, or capability.
title: '[Feature] Short description of the feature'
labels: 'feature request'
---

# Motivation (Required)
<!-- Why do we need this? What is the current workaround? -->
- **Problem:** (1 sentence description of the pain point).
- **Solution:** (1 sentence description of why this feature helps).
- **Prioritization:** Critical | High | Medium | Nice-to-Have

# Proposed API (Recommended)
<!-- Show how you want the feature to be called/used. Keep it simple. -->
```python
import auramask

# Example of how you expect the API to work
result = SomeNewMethod(...)
# or
model = SomeNewModel(config={})
```

## Current Backend Compatibility (Required)
<!-- **Keras 3 Core Philosophy:** The API should ideally work across all backends. -->
 TensorFlow (Fully supported)
 PyTorch (Fully supported)
 JAX (Fully supported)
 Other (Specify and justify why)
Note: If a feature cannot be implemented on all backends, it must be marked as experimental or restricted to specific backend directories (e.g., keras.src/backend/tensorflow).

## Implementation Constraints
<!-- What might prevent this from working? -->
- Heavy memory consumption
- Complex mathematical operation
- Requires custom C/C++ extensions
- Not supported by underlying ops libraries

## Alternatives Tried
<!-- Show what you have already attempted -->
- Searched the [Documentation](https://gitlab.com/raccs-lab/auramask-library/auramask/-/wikis/home)
- Used an external library/workaround

> **Before requesting:**
> - Check our [Feature Roadmap](#) (Public).
> - We prioritize backend-agnostic features first.
> - Features requiring custom C++ code or breaking API changes are rarely accepted without a strong consensus.