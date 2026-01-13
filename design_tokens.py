"""
Qwen Image Generator - Design System

Design Direction: Precision & Density + Data Analysis
- Dark mode, cool slate foundation
- Borders-only depth (minimal shadows)
- Monospace for data values
- 4px grid spacing
- Indigo accent for actions

Usage:
    from design_tokens import COLORS, SPACING, CSS_VARS
"""

# 4px Grid Spacing System
SPACING = {
    'micro': 4,      # Icon gaps
    'tight': 8,      # Within components
    'standard': 12,  # Between related elements
    'comfortable': 16,  # Section padding
    'generous': 24,  # Between sections
    'major': 32,     # Major separation
}

# Color System - Cool Slate Foundation
COLORS = {
    # Backgrounds (dark to light)
    'bg-base': '#0f1117',
    'bg-surface': '#151821',
    'bg-elevated': '#1c1f2b',
    'bg-hover': '#252937',

    # Borders
    'border': 'rgba(255, 255, 255, 0.08)',
    'border-subtle': 'rgba(255, 255, 255, 0.05)',
    'border-hex': '#1f2330',

    # Text hierarchy (4 levels)
    'fg-primary': '#f0f1f3',
    'fg-secondary': '#a0a4af',
    'fg-muted': '#6b7080',
    'fg-faint': '#464b5c',

    # Accent (Indigo - creativity, AI-forward)
    'accent': '#6366f1',
    'accent-hover': '#7577f5',
    'accent-muted': 'rgba(99, 102, 241, 0.15)',
    'accent-subtle': 'rgba(99, 102, 241, 0.08)',

    # Semantic (slightly desaturated for dark mode)
    'success': '#22c55e',
    'success-muted': 'rgba(34, 197, 94, 0.15)',
    'warning': '#eab308',
    'warning-muted': 'rgba(234, 179, 8, 0.15)',
    'error': '#ef4444',
    'error-muted': 'rgba(239, 68, 68, 0.15)',
}

# Typography
TYPOGRAPHY = {
    'font-sans': '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif',
    'font-mono': '"SF Mono", "Fira Code", "JetBrains Mono", Consolas, monospace',

    # Scale (tight hierarchy)
    'text-xs': '11px',
    'text-sm': '12px',
    'text-base': '13px',
    'text-md': '14px',
    'text-lg': '16px',
    'text-xl': '18px',
    'text-2xl': '24px',

    # Weights
    'font-normal': '400',
    'font-medium': '500',
    'font-semibold': '600',
    'font-bold': '700',
}

# Border Radius (sharp corners for technical feel)
RADIUS = {
    'sm': '4px',
    'md': '6px',
    'lg': '8px',
    'xl': '12px',
}

# Animation
ANIMATION = {
    'duration-fast': '150ms',
    'duration-base': '200ms',
    'easing': 'cubic-bezier(0.25, 1, 0.5, 1)',
}

# CSS Variables for HTML embedding
CSS_VARS = f"""
:root {{
    /* Spacing */
    --space-micro: {SPACING['micro']}px;
    --space-tight: {SPACING['tight']}px;
    --space-standard: {SPACING['standard']}px;
    --space-comfortable: {SPACING['comfortable']}px;
    --space-generous: {SPACING['generous']}px;
    --space-major: {SPACING['major']}px;

    /* Colors - Backgrounds */
    --bg-base: {COLORS['bg-base']};
    --bg-surface: {COLORS['bg-surface']};
    --bg-elevated: {COLORS['bg-elevated']};
    --bg-hover: {COLORS['bg-hover']};

    /* Colors - Borders */
    --border: {COLORS['border']};
    --border-subtle: {COLORS['border-subtle']};

    /* Colors - Text */
    --fg-primary: {COLORS['fg-primary']};
    --fg-secondary: {COLORS['fg-secondary']};
    --fg-muted: {COLORS['fg-muted']};
    --fg-faint: {COLORS['fg-faint']};

    /* Colors - Accent */
    --accent: {COLORS['accent']};
    --accent-hover: {COLORS['accent-hover']};
    --accent-muted: {COLORS['accent-muted']};

    /* Colors - Semantic */
    --success: {COLORS['success']};
    --success-muted: {COLORS['success-muted']};
    --warning: {COLORS['warning']};
    --warning-muted: {COLORS['warning-muted']};
    --error: {COLORS['error']};
    --error-muted: {COLORS['error-muted']};

    /* Typography */
    --font-sans: {TYPOGRAPHY['font-sans']};
    --font-mono: {TYPOGRAPHY['font-mono']};

    /* Radius */
    --radius-sm: {RADIUS['sm']};
    --radius-md: {RADIUS['md']};
    --radius-lg: {RADIUS['lg']};

    /* Animation */
    --duration-fast: {ANIMATION['duration-fast']};
    --duration-base: {ANIMATION['duration-base']};
    --easing: {ANIMATION['easing']};
}}
"""

# Base CSS reset and utilities
CSS_BASE = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: var(--font-sans);
    font-size: 13px;
    line-height: 1.5;
    color: var(--fg-primary);
    background: var(--bg-base);
    -webkit-font-smoothing: antialiased;
}

/* Typography utilities */
.text-primary { color: var(--fg-primary); }
.text-secondary { color: var(--fg-secondary); }
.text-muted { color: var(--fg-muted); }
.text-faint { color: var(--fg-faint); }

.font-mono { font-family: var(--font-mono); }
.font-medium { font-weight: 500; }
.font-semibold { font-weight: 600; }
.font-bold { font-weight: 700; }

.text-xs { font-size: 11px; }
.text-sm { font-size: 12px; }
.text-base { font-size: 13px; }
.text-lg { font-size: 16px; }
.text-xl { font-size: 18px; }

/* Monospace for data */
.tabular-nums { font-variant-numeric: tabular-nums; }

/* Surface components */
.surface {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
}

.elevated {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
}

/* Button base */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-tight);
    padding: var(--space-tight) var(--space-comfortable);
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 600;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all var(--duration-fast) var(--easing);
    white-space: nowrap;
}

.btn-primary {
    background: var(--accent);
    color: white;
}
.btn-primary:hover {
    background: var(--accent-hover);
}

.btn-secondary {
    background: var(--bg-elevated);
    color: var(--fg-secondary);
    border: 1px solid var(--border);
}
.btn-secondary:hover {
    background: var(--bg-hover);
    color: var(--fg-primary);
}

/* Input base */
.input {
    width: 100%;
    padding: var(--space-tight) var(--space-standard);
    font-family: var(--font-sans);
    font-size: 13px;
    color: var(--fg-primary);
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    outline: none;
    transition: border-color var(--duration-fast) var(--easing);
}
.input:focus {
    border-color: var(--accent);
}
.input::placeholder {
    color: var(--fg-muted);
}

/* Label */
.label {
    display: block;
    font-size: 11px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: var(--space-tight);
}
"""


def get_full_css():
    """Return complete CSS including variables and base styles"""
    return CSS_VARS + CSS_BASE
