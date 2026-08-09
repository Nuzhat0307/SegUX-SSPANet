import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  FileText,
  History,
  Settings,
  LogOut,
  Brain,
  Activity,
} from 'lucide-react'
import { useAuth } from '../lib/auth'

const navItems = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/upload', label: 'Upload MRI', icon: Upload },
  { to: '/history', label: 'History', icon: History },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/settings', label: 'Settings', icon: Settings },
]

export default function Sidebar() {
  const { user, signOut } = useAuth()

  return (
    <aside className="fixed left-0 top-0 z-30 flex h-screen w-64 flex-col border-r border-neutral-200 bg-white">
      <div className="flex items-center gap-3 border-b border-neutral-200 px-5 py-5">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-600">
          <Brain className="h-6 w-6 text-white" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-neutral-900">SegUX-SSPANet</h1>
          <p className="text-xs text-neutral-500">Brain Tumor Diagnosis</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'sidebar-link-active' : ''}`
              }
            >
              <Icon className="h-5 w-5" />
              <span>{item.label}</span>
            </NavLink>
          )
        })}
      </nav>

      <div className="border-t border-neutral-200 p-3">
        <div className="mb-3 flex items-center gap-3 rounded-lg bg-neutral-50 px-3 py-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-100">
            <Activity className="h-4 w-4 text-primary-600" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-semibold text-neutral-700">
              {user?.email}
            </p>
            <p className="text-xs text-neutral-400">Clinician</p>
          </div>
        </div>
        <button
          onClick={() => signOut()}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-neutral-600 transition-all hover:bg-error-50 hover:text-error-600"
        >
          <LogOut className="h-5 w-5" />
          <span>Sign Out</span>
        </button>
      </div>
    </aside>
  )
}
