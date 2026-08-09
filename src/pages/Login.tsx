import { useState, FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { Brain, Mail, Lock, AlertCircle, Loader2 } from 'lucide-react'

export default function Login() {
  const { signIn } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    const { error } = await signIn(email, password)
    setLoading(false)
    if (error) {
      setError(error)
    } else {
      navigate('/dashboard')
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-50 px-4">
      <div className="grid-pattern absolute inset-0 -z-10" />
      <div className="w-full max-w-md animate-slide-up">
        <div className="mb-8 flex flex-col items-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary-600 shadow-lg shadow-primary-600/20">
            <Brain className="h-9 w-9 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-neutral-900">SegUX-SSPANet</h1>
          <p className="mt-1 text-sm text-neutral-500">Brain Tumor Diagnosis System</p>
        </div>

        <div className="card p-8">
          <h2 className="mb-1 text-lg font-semibold text-neutral-900">Sign in to your account</h2>
          <p className="mb-6 text-sm text-neutral-500">
            Enter your credentials to access the diagnostic platform
          </p>

          {error && (
            <div className="mb-4 flex items-center gap-2 rounded-lg bg-error-50 px-3 py-2.5 text-sm text-error-700">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="label-text" htmlFor="email">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="input-field pl-10"
                  placeholder="you@hospital.com"
                  required
                />
              </div>
            </div>

            <div>
              <label className="label-text" htmlFor="password">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="input-field pl-10"
                  placeholder="Enter your password"
                  required
                />
              </div>
            </div>

            <button type="submit" disabled={loading} className="btn-primary w-full">
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-neutral-500">
            Don't have an account?{' '}
            <Link to="/register" className="font-semibold text-primary-600 hover:text-primary-700">
              Create one
            </Link>
          </p>
        </div>

        <p className="mt-6 text-center text-xs text-neutral-400">
          For research and educational purposes. Not a substitute for professional medical diagnosis.
        </p>
      </div>
    </div>
  )
}
