import baseConfig from '@a/eslint-config/base'
import restrictEnvAccess from '@a/eslint-config/restrict-env'
import { defineConfig } from 'eslint/config'

export default defineConfig({ ignores: ['dist/**', '.venv/**'] }, baseConfig, restrictEnvAccess)
