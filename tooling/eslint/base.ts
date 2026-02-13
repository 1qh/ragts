/// <reference types="./types.d.ts" />

import eslintReact from '@eslint-react/eslint-plugin'
import { includeIgnoreFile } from '@eslint/compat'
import eslint from '@eslint/js'
import importPlugin from 'eslint-plugin-import'
import { configs as perfectionist } from 'eslint-plugin-perfectionist'
import preferArrow from 'eslint-plugin-prefer-arrow-functions'
import turbo from 'eslint-plugin-turbo'
import { defineConfig, globalIgnores } from 'eslint/config'
import * as path from 'node:path'
import tseslint from 'typescript-eslint'

export default defineConfig(
  includeIgnoreFile(path.join(import.meta.dirname, '../../.gitignore')),
  globalIgnores(['convex/_generated']),
  perfectionist['recommended-natural'],
  { ignores: ['postcss.config.mjs'] },
  {
    extends: [
      eslint.configs.recommended,
      eslint.configs.all,
      ...tseslint.configs.all,
      ...tseslint.configs.recommended,
      ...tseslint.configs.recommendedTypeChecked,
      ...tseslint.configs.stylisticTypeChecked,
      eslintReact.configs['recommended-type-checked'],
      eslintReact.configs.recommended
    ],
    files: ['**/*.js', '**/*.ts', '**/*.tsx'],
    plugins: {
      import: importPlugin,
      preferArrow,
      turbo
    },
    rules: {
      '@eslint-react/avoid-shorthand-boolean': 'off',
      '@eslint-react/avoid-shorthand-fragment': 'off',
      '@eslint-react/no-implicit-key': 'warn',
      '@eslint-react/no-missing-context-display-name': 'off',
      '@typescript-eslint/consistent-return': 'off',
      '@typescript-eslint/consistent-type-imports': [
        'warn',
        { fixStyle: 'separate-type-imports', prefer: 'type-imports' }
      ],
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      '@typescript-eslint/init-declarations': 'off',
      '@typescript-eslint/naming-convention': [
        'warn',
        { format: ['camelCase', 'UPPER_CASE', 'PascalCase'], selector: 'variable' }
      ],
      '@typescript-eslint/no-confusing-void-expression': 'off',
      '@typescript-eslint/no-floating-promises': 'off',
      '@typescript-eslint/no-magic-numbers': ['warn', { ignore: [-1, 0, 1, 2, 100] }],
      '@typescript-eslint/no-misused-promises': [2, { checksVoidReturn: { attributes: false } }],
      '@typescript-eslint/no-non-null-assertion': 'error',
      '@typescript-eslint/no-unnecessary-condition': ['error', { allowConstantLoopConditions: true }],
      '@typescript-eslint/no-unsafe-type-assertion': 'off',
      '@typescript-eslint/prefer-readonly-parameter-types': 'off',
      '@typescript-eslint/strict-boolean-expressions': 'off',
      camelcase: 'off',
      'capitalized-comments': ['error', 'always', { ignorePattern: 'oxlint|biome|console|let|const|return|if|for|throw' }],
      curly: ['error', 'multi'],
      'id-length': 'off',
      'import/consistent-type-specifier-style': ['error', 'prefer-top-level'],
      'max-lines': 'off',
      'max-lines-per-function': 'off',
      'new-cap': ['error', { capIsNewExceptionPattern: 'Inter|JetBrains_Mono' }],
      'no-console': 'warn',
      'no-duplicate-imports': ['error', { allowSeparateTypeImports: true }],
      'no-nested-ternary': 'off',
      'no-promise-executor-return': 'off',
      'no-ternary': 'off',
      'no-undefined': 'off',
      'no-underscore-dangle': 'off',
      'one-var': ['warn', 'consecutive'],
      'perfectionist/sort-variable-declarations': 'off',
      'preferArrow/prefer-arrow-functions': ['error', { returnStyle: 'implicit' }],
      'sort-imports': 'off',
      'sort-keys': 'off',
      'sort-vars': ['off', { ignoreCase: true }]
    }
  },
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname
      }
    },
    linterOptions: { reportUnusedDisableDirectives: true }
  }
)
