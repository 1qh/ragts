import type { Linter } from 'eslint'

import eslintPluginBetterTailwindcss from 'eslint-plugin-better-tailwindcss'
import reactPlugin from 'eslint-plugin-react'
import reactHooks from 'eslint-plugin-react-hooks'
import { defineConfig } from 'eslint/config'
import * as path from 'node:path'

export default defineConfig(reactHooks.configs.flat['recommended-latest'] as { rules: Linter.RulesRecord }, {
  files: ['**/*.ts', '**/*.tsx'],
  ...reactPlugin.configs.flat.all,
  ...reactPlugin.configs.flat['jsx-runtime'],
  languageOptions: {
    ...reactPlugin.configs.flat.all?.languageOptions,
    ...reactPlugin.configs.flat['jsx-runtime']?.languageOptions,
    globals: {
      React: 'writable'
    }
  },
  plugins: {
    'better-tailwindcss': eslintPluginBetterTailwindcss,
    react: reactPlugin
  },
  rules: {
    ...reactPlugin.configs['jsx-runtime'].rules,
    ...reactPlugin.configs.all.rules,
    ...eslintPluginBetterTailwindcss.configs['recommended-error'].rules,
    'better-tailwindcss/enforce-canonical-classes': 'error',
    'better-tailwindcss/enforce-consistent-important-position': 'error',
    'better-tailwindcss/enforce-consistent-line-wrapping': 'off',
    'better-tailwindcss/enforce-consistent-variable-syntax': 'error',
    'better-tailwindcss/enforce-shorthand-classes': 'error',
    'better-tailwindcss/no-deprecated-classes': 'error',
    'better-tailwindcss/no-unknown-classes': ['warn', { ignore: ['group', 'peer', 'nodrag', 'nopan', 'nowheel'] }],
    'react-hooks/preserve-manual-memoization': 'warn',
    'react-hooks/set-state-in-effect': 'warn',
    'react/button-has-type': 'off',
    'react/forbid-component-props': 'off',
    'react/function-component-definition': 'off',
    'react/hook-use-state': 'off',
    'react/jsx-child-element-spacing': 'off',
    'react/jsx-closing-bracket-location': 'off',
    'react/jsx-curly-newline': 'off',
    'react/jsx-filename-extension': ['error', { extensions: ['.tsx'] }],
    'react/jsx-handler-names': 'off',
    'react/jsx-indent': 'off',
    'react/jsx-indent-props': 'off',
    'react/jsx-max-depth': 'off',
    'react/jsx-max-props-per-line': 'off',
    'react/jsx-newline': 'off',
    'react/jsx-no-bind': 'off',
    'react/jsx-no-constructed-context-values': 'off',
    'react/jsx-no-literals': 'off',
    'react/jsx-one-expression-per-line': 'off',
    'react/jsx-pascal-case': 'off',
    'react/jsx-props-no-spreading': 'off',
    'react/jsx-sort-props': ['error', { ignoreCase: true }],
    'react/no-array-index-key': 'off',
    'react/no-danger': 'off',
    'react/no-multi-comp': 'off',
    'react/no-unstable-nested-components': 'off',
    'react/prefer-read-only-props': 'off',
    'react/prop-types': 'off',
    'react/require-default-props': 'off'
  },
  settings: {
    'better-tailwindcss': {
      entryPoint: path.join(import.meta.dirname, '../../packages/ui/src/styles/globals.css')
    }
  }
})
