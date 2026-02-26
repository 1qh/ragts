import { defineConfig } from 'eslint/config'

export default defineConfig(
  { ignores: ['**/env.ts'] },
  {
    files: ['**/*.js', '**/*.ts', '**/*.tsx'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          importNames: ['env'],
          message: "Use `import env from '~/env'` instead to ensure validated types.",
          name: 'process'
        }
      ],
      'no-restricted-properties': [
        'error',
        {
          message: "Use `import env from '~/env'` instead to ensure validated types.",
          object: 'process',
          property: 'env'
        }
      ]
    }
  }
)
