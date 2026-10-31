# Project Agent Guidelines

Follow these guidelines when working in this repository.

## 1. Reuse Existing Logic

- Search for existing implementations before adding a function, helper, or component.
- Do not create another function that performs the same job. Reuse or extend the existing implementation, or extract shared logic when necessary.
- Keep a single implementation of each business rule so all callers produce consistent results.
- Example: The illustration prompt builder must use the same prompt-building function for preview and submission. Preview displays its result; submission sends its result. Do not maintain separate builders with duplicated assembly logic.

## 2. Create Tests Only When Necessary

- Add tests only when they are needed to verify meaningful behavior, prevent a concrete regression, or cover a failure that is difficult to check reliably by hand.
- Do not add tests, tester scripts, or test infrastructure by default for every change.
- Example: Do not create a dedicated tester merely to check a character count.
- Prefer relevant existing tests and focused manual checks when they are sufficient. Keep any necessary new test limited to the behavior at risk.
- Report what was actually checked. Do not claim that tests ran when they did not.

## 3. Communicate Using the Actual UI Names

- Explain changes and instructions in language the user can understand.
- Use the exact names shown in the current UI for pages, tabs, buttons, fields, and settings. Check the UI or its source before referring to a label.
- Do not invent feature names, substitute internal identifiers for visible labels, or rename controls in an explanation.
- When giving navigation instructions, use the actual sequence of UI labels so the user can follow it.
- Keep technical details focused on what helps the user understand the result or take the next action.

## 4. Keep Function and Variable Names Consistent

- Use one canonical name for the same concept or value throughout the code. Do not introduce synonyms, alternate abbreviations, or different names for the same variable as it moves between functions or layers.
- Follow the language's naming conventions while keeping the same terminology, such as the same words in Python snake_case and JavaScript camelCase.
- Name functions consistently with the feature and action they implement. Distinct actions may have distinct names, but must use the same term for their shared concept.
- When a UI feature, field, or action is renamed, update its corresponding function names, variable names, and references in the same change. Update related comments and documentation as well.
- For localized UI labels, use a consistent English equivalent in code and the actual displayed label in user-facing explanations.

## 5. Build Responsive UI Using shadcn/ui References

- Use [shadcn/ui](https://ui.shadcn.com/) as a reference for component design, spacing, typography, interaction, and responsive layouts.
- Make layouts usable on desktop, tablet, and mobile screens. Let columns stack, controls wrap, and navigation collapse when space is limited. Keep primary actions reachable and avoid unintended horizontal scrolling.
- Preserve clear labels, keyboard navigation, visible focus states, and useful loading, empty, and error states.
- Reuse the project's existing frontend components and styles. Adapt shadcn/ui patterns to the current frontend rather than introducing a framework migration solely to copy a template.
- When a useful template is found, add its name, source URL, intended use, and relevant adaptation notes to the references below. Keep these references current as the UI evolves.

### UI Templates and Component References

| Reference | Intended use | Adaptation notes |
| --- | --- | --- |
| [shadcn/ui Blocks - dashboard-01](https://ui.shadcn.com/blocks#dashboard-01) | Reference for an application layout with sidebar navigation and a main content area. | Reuse relevant layout patterns with the project's actual UI labels; include only sections needed by the feature. |
| [Sidebar](https://ui.shadcn.com/docs/components/sidebar) | Reference for navigation that adapts to available screen space. | Keep navigation names consistent across desktop and mobile layouts. |
| [Field](https://ui.shadcn.com/docs/components/field) | Reference for labeled settings, helper text, and validation messages. | Group related settings and adapt field arrangements to narrow screens. |
