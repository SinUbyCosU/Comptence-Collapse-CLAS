# Contributing to Competence Collapse Research

Thank you for your interest in contributing to this research! We welcome contributions in several forms:

## Ways to Contribute

### 1. 🐛 Bug Reports and Issues

If you find any issues with:
- Code reproducibility
- Data access or loading
- Documentation errors
- Calculation discrepancies

Please [open an issue](https://github.com/SinUbyCosU/competence-collapse/issues) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version, dependency versions)

### 2. 📝 Documentation Improvements

Help us improve:
- Clarify confusing explanations
- Fix typos and grammar
- Add examples or tutorials
- Improve code comments
- Translate documentation

### 3. 🔬 Extensions and Replications

We encourage researchers to:
- Test CLAS on additional language pairs
- Evaluate on different model architectures
- Explore alternative steering methods
- Conduct follow-up studies

**When publishing extensions**:
- Please cite the original paper
- Clearly indicate which components are novel
- Share your code and data when possible

### 4. 💻 Code Contributions

We welcome:
- Performance optimizations
- Additional evaluation metrics
- Visualization improvements
- New analysis scripts
- Bug fixes

## Contribution Process

### Step 1: Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/competence-collapse.git
cd competence-collapse
git remote add upstream https://github.com/SinUbyCosU/competence-collapse.git
```

### Step 2: Create a Branch

```bash
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/issue-description
```

### Step 3: Make Changes

- Follow existing code style
- Add comments for complex logic
- Update documentation if needed
- Test your changes thoroughly

### Step 4: Commit

```bash
git add .
git commit -m "Clear description of changes"
```

**Commit message guidelines**:
- Use present tense ("Add feature" not "Added feature")
- Be specific and concise
- Reference issues when applicable (#123)

### Step 5: Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title and description
- Reference to related issues
- Summary of changes
- Test results (if applicable)

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints where possible
- Maximum line length: 100 characters
- Use meaningful variable names

```python
# Good
def compute_instructional_density(text: str, lambda_weight: float = 0.5) -> float:
    """Compute Instructional Density metric."""
    pass

# Avoid
def calc_d(t, l=0.5):
    pass
```

### Documentation

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include examples in docstrings
- Keep README and guides up-to-date

```python
def apply_steering(model, vector, alpha=1.0):
    """
    Apply CLAS steering to model activations.
    
    Args:
        model: HuggingFace transformer model
        vector: Pre-computed steering vector (torch.Tensor)
        alpha: Steering coefficient, range [0, 2.0]
    
    Returns:
        Hook handle that can be removed with handle.remove()
    
    Example:
        >>> vec = torch.load('vectors/pc1_layer_16.pt')
        >>> hook = apply_steering(model, vec, alpha=1.0)
        >>> # ... generate text ...
        >>> hook.remove()
    """
    pass
```

## Research Ethics and Best Practices

### When Extending This Work

1. **Cite Appropriately**: Always cite the original paper and repository
2. **Acknowledge Limitations**: Be transparent about what changed
3. **Share Openly**: Make your extensions available when possible
4. **Respect Privacy**: If using human evaluation, obtain proper consent

### Sensitive Content

This repository contains:
- **Adversarial prompts** (for safety testing)
- **Medical scenarios** (for linguistic analysis only)

**Guidelines**:
- Do not use medical prompts for actual medical advice
- Handle adversarial content responsibly
- Consider ethical implications of your extensions

### Data Usage

- Follow the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license for data
- Cite the paper when using data
- Respect annotator contributions
- Be transparent about data modifications

## Review Process

1. **Automated Checks**: CI/CD runs basic tests
2. **Code Review**: Maintainer reviews for:
   - Code quality and style
   - Documentation completeness
   - Reproducibility
   - Alignment with project goals
3. **Discussion**: Open dialogue on design decisions
4. **Approval**: Merge when ready

**Timeline**: Expect 3-7 days for initial review

## Recognition

Contributors will be:
- Listed in repository contributors
- Acknowledged in release notes
- Credited in derivative works (as appropriate)

Substantial contributions may warrant co-authorship in follow-up publications.

## Questions?

- **Technical issues**: [Open an issue](https://github.com/SinUbyCosU/competence-collapse/issues)
- **Research collaboration**: Email yadav23@iiserb.ac.in
- **General questions**: Use [GitHub Discussions](https://github.com/SinUbyCosU/competence-collapse/discussions)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. We expect all contributors to:

- Be respectful and professional
- Welcome diverse perspectives
- Focus on constructive feedback
- Assume good intentions
- Respect privacy and confidentiality

**Unacceptable behavior includes**:
- Harassment or discrimination
- Personal attacks
- Publishing private information
- Plagiarism or misrepresentation

**Enforcement**: Violations may result in temporary or permanent ban from the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License (for code) and CC BY 4.0 (for data/documentation).

---

Thank you for helping advance multilingual NLP research! 🌍
