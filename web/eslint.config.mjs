import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

// Spec §4.5 "Theming": theme is expressed as design tokens (CSS custom
// properties), never as literal colours inside components. This rule fails
// the build on any hex colour literal (string or template literal) found in
// component/app source so the rule holds up over time instead of relying on
// convention. Colour tokens themselves are defined once, in globals.css,
// which this rule does not touch.
const HEX_COLOR_PATTERN =
  "/#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\\b/";

const noHardcodedColor = {
  files: ["src/components/**/*.{ts,tsx}", "src/app/**/*.{ts,tsx}"],
  ignores: ["**/*.test.{ts,tsx}"],
  rules: {
    "no-restricted-syntax": [
      "error",
      {
        selector: `Literal[value=${HEX_COLOR_PATTERN}]`,
        message:
          "No hardcoded hex colours in components. Use a design token (CSS custom property, e.g. var(--color-accent)) defined in globals.css instead. See spec §4.5 Theming.",
      },
      {
        selector: `TemplateElement[value.raw=${HEX_COLOR_PATTERN}]`,
        message:
          "No hardcoded hex colours in components. Use a design token (CSS custom property, e.g. var(--color-accent)) defined in globals.css instead. See spec §4.5 Theming.",
      },
    ],
  },
};

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  noHardcodedColor,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Throwaway design comps (gitignored). Deliberately plain HTML/JS with
    // hardcoded colours -- linting them as app source is noise.
    "mockups/**",
  ]),
]);

export default eslintConfig;
