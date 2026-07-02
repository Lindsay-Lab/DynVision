---
title: Developer Guide
---

Welcome to the DynVision Developer Guide! This section provides comprehensive resources for contributors, maintainers, and AI assistants (like Claude Code) working on the DynVision codebase.

## Purpose

The Developer Guide serves multiple audiences:

- **Contributors**: Understand architecture, follow coding standards, learn workflows
- **Maintainers**: Track TODOs, plan features, manage dependencies
- **AI Assistants**: Comprehensive context for Claude Code and similar tools
- **Researchers**: Understand design decisions and biological motivations

## Organization

The Developer Guide is organized into four main sections:

### 📋 [Planning](planning/todo-docs.md)

Forward-looking documentation tracking what needs to be done:

- **[Documentation TODOs](planning/todo-docs.md)**: Documentation fixes and doc/code mismatches (24 issues)
  - Critical issues (project naming, broken links, class mismatches)
  - Documentation gaps (missing examples, images, templates)
  - Code vs documentation mismatches
  - Performance and optimization docs needed

- **[Development Roadmap](planning/todo-roadmap.md)**: Feature development and enhancements (30 items)
  - Phase 1: Quick Wins (user onboarding, config validation, pre-commit hooks)
  - Phase 2: Foundation (test suite, CI/CD, minimal examples)
  - Phase 3: Tools & Analysis (inspector, benchmarking, profiling)
  - Phase 4: Advanced Features (performance optimizations, research tools)

**When to use**: Before starting new work, check these files to avoid duplication and align with project priorities.

---

### 📖 [Guides](guides/ai-style-guide.md)

How-to information for developers and AI assistants:

- **[AI Style Guide](guides/ai-style-guide.md)**: **START HERE** - Core principles for AI-assisted research software development
  - Research software best practices (scientific correctness, reproducibility, performance)
  - Investigation → Analysis → Implementation workflow
  - Code organization, documentation, testing, error handling
  - Communication and collaboration guidelines
  - General principles applicable to any research software project
  - **Use this to prime AI assistants before working on any task**

- **[Claude Code Guide](guides/claude-guide.md)**: DynVision-specific context for Claude Code (formerly CLAUDE.md)
  - Project overview and design philosophy
  - Development commands (setup, running experiments, code quality)
  - Detailed architecture (multi-inheritance, base classes, components)
  - Configuration system and parameter handling
  - Common workflows (adding models, experiments, recurrent connections)
  - Known issues and quick reference
  - **Use this after the AI Style Guide for project-specific details**

- **[Documentation Style Guide](guides/documentation-style.md)**: Standards for writing documentation
  - Documentation philosophy and structure
  - Writing style and formatting
  - Code examples and API documentation
  - Diagrams and visual elements
  - Review checklist

- **[Research Software](guides/research-software.md)**: Considerations for Software Dev in Research
  - Expert research software developer agent persona
  - Code review and analysis framework (scientific correctness, architecture, performance, quality, documentation)
  - Structured deliverables format (executive summary, detailed analysis, code examples, roadmap)
  - Specialized areas (scientific computing optimization, neural networks, computational neuroscience)
  - Software engineering best practices (testing, documentation, reproducibility, sustainability)
  - Project-specific applications and context adaptation

- **[Software Patterns](guides/software-patterns.md)**: Design patterns for scientific computing
  - Architectural patterns (Layered, Pipeline, Domain-Driven Design, Event-Driven)
  - Creational patterns (Factory, Abstract Factory, Builder, Singleton, Prototype)
  - Structural patterns (Adapter, Facade, Composite, Decorator, Bridge)
  - Behavioral patterns (Strategy, Observer, Command, Template Method, State, Iterator, Visitor)
  - Scientific computing patterns (Computation Graph, Lazy Evaluation, Parameter Management, Data Pipeline, Experiment Tracking)

**When to use**:

- **AI Assistants**: Start with AI Style Guide → Claude Code Guide → other guides as needed
- **Human Developers**: Reference guides when developing features, writing docs, or onboarding

---

### 🔧 [Dependencies](dependencies/snakemake.md)

Knowledge about external frameworks and libraries:

- **[Snakemake](dependencies/snakemake.md)**: Workflow management system
  - Fundamentals of Snakemake (rules, wildcards, config)
  - DynVision workflow structure
  - Writing new rules and experiments
  - Debugging workflows
  - Cluster integration

- **[PyTorch Lightning](dependencies/pytorch-lightning.md)**: Deep learning framework
  - Lightning integration in DynVision
  - LightningModule structure
  - Training loops and callbacks
  - Logging and checkpointing
  - Multi-GPU training

- **[FFCV](dependencies/ffcv.md)**: Fast data loading library
  - What FFCV is and when to use it
  - Installation and setup
  - Creating .beton files
  - Performance considerations
  - Troubleshooting

**When to use**: Learning about framework capabilities, debugging integration issues, or optimizing performance.

---

## Navigation Tips

### For New Contributors

Recommended reading order:

1. Start with [Claude Code Guide](guides/claude-guide.md) for project overview
2. Reference [Research Software](guides/software-patterns.md) for design guidance
3. Consult [Documentation Style Guide](guides/documentation-style.md) when writing docs
4. Review [Documentation TODOs](planning/todo-docs.md) to find contribution opportunities
5. Check [Development Roadmap](planning/todo-roadmap.md) for aligned feature work

### For AI Assistants

**Required Reading Order**:

1. **[AI Style Guide](guides/ai-style-guide.md)** first - Establishes core principles for research software development
2. **[Claude Code Guide](guides/claude-guide.md)** second - Provides DynVision-specific architecture and conventions

The AI Style Guide teaches you **how to approach** research software tasks with emphasis on:

- Scientific correctness and reproducibility
- Investigation → Analysis → Implementation workflow
- Performance optimization strategies
- Documentation and testing standards
- Communication with researchers

The Claude Code Guide provides **project-specific context**:

- Complete architecture with inheritance diagrams
- All parameter aliases and conventions
- Common workflows with examples
- Known issues and inconsistencies

Together, these guides minimize the need for extensive code reading while ensuring accuracy and adherence to best practices.

### For Maintainers

Track project health via:

- [Documentation TODOs](planning/todo-docs.md) - 24 documentation issues to address
- [Development Roadmap](planning/todo-roadmap.md) - 30 development items organized by priority

Both files include priority rankings and effort estimates.

## Quick Links

**Most Referenced**:

- [Claude Code Guide](guides/claude-guide.md) - Complete developer reference
- [Todo Docs](planning/todo-docs.md) - Known documentation issues
- [Roadmap](planning/todo-roadmap.md) - Planned features

**Dependency Docs**:

- [Snakemake Patterns](dependencies/snakemake.md) - Workflow management
- [PyTorch Lightning](dependencies/pytorch-lightning.md) - Training framework
- [FFCV Integration](dependencies/ffcv.md) - Fast data loading

- **[Software Patterns](guides/software-patterns.md)**: Design patterns for scientific computing
  - Architectural patterns, creational/structural/behavioral patterns, scientific computing patterns
  - **When to use**: Design guidance for new components

## Contributing Workflow

1. **Find a Task**: Check [Documentation TODOs](planning/todo-docs.md) or [Development Roadmap](planning/todo-roadmap.md)
2. **Understand Context**: Read [Claude Code Guide](guides/claude-guide.md) architecture section
3. **Follow Standards**: Reference [Documentation Style Guide](guides/documentation-style.md)
4. **Implement**: Use codebase patterns for guidance
5. **Test**: Add tests (see Roadmap #29-#31 for test infrastructure plans)
6. **Document**: Update relevant docs following style guide
7. **Review**: Check against [Claude Code Guide](guides/claude-guide.md) for consistency

## Keeping Documentation Current

This Developer Guide should be updated:

- **When adding features**: Update [Claude Code Guide](guides/claude-guide.md) architecture
- **When finding bugs**: Add to [Documentation TODOs](planning/todo-docs.md)
- **When planning work**: Update [Development Roadmap](planning/todo-roadmap.md)
- **When writing docs**: Follow [Documentation Style Guide](guides/documentation-style.md)
- **When changing dependencies**: Update relevant dependency docs

## Related Resources

- **[Main Documentation](../index.md)**: User-facing documentation
- **[Contributing Guide](../contributing.md)**: How to contribute
- **[Getting Started](../getting-started.md)**: First steps with DynVision
- **[User Guide](../user-guide/index.md)**: Task-oriented guides
- **[Reference](../reference/index.md)**: API and component reference

## Questions or Feedback?

- Check [Documentation TODOs](planning/todo-docs.md) to see if your question is a known issue
- Open a GitHub issue for new documentation needs
- Contact maintainers: robin.gutzen@nyu.edu

---

**Last Updated**: 2025-10-23

This Developer Guide is maintained by the DynVision team and is designed to evolve with the project. Contributions and suggestions are welcome!
