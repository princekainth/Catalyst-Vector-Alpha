# Catalyst Vector Alpha Enhanced Dashboard

A modern React-based dashboard for monitoring and managing the Catalyst Vector Alpha autonomous SRE platform.

## 🚀 Features

- **Real-time System Monitoring**: Live health status, metrics, and performance data
- **Agent Management**: View and control all agents in the swarm
- **Task Tracking**: Monitor recent tasks and their execution status
- **Pending Approvals**: Human-in-the-loop approval system for critical operations
- **Comprehensive Metrics**: CPU, memory, agent performance, and system health
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Theme**: Optimized for 24/7 operations centers

## 📁 Installation

### Prerequisites

- Node.js v16+ (recommended v18+)
- npm v7+ (comes with Node.js)
- Python 3.8+ (for the backend)

### Setup

1. **Install dependencies:**
   ```bash
   cd dashboard
   npm install
   ```

2. **Build the dashboard:**
   ```bash
   npm run build
   ```

3. **Or use the build script:**
   ```bash
   ./build_dashboard.sh
   ```

## 🔧 Configuration

The dashboard connects to the Flask backend API. You can configure the API base URL in:

- `src/api.js` - Change the `baseURL` to point to your backend
- Environment variable: `REACT_APP_API_URL`

## 📂 Project Structure

```
dashboard/
├── public/              # Static assets and HTML template
├── src/                 # React source code
│   ├── api/             # API client and services
│   ├── components/      # Reusable UI components
│   ├── pages/           # Dashboard pages
│   ├── App.js           # Main application router
│   ├── index.js         # Entry point
│   └── index.css        # Global styles
├── build_dashboard.sh   # Build script
└── package.json         # Dependencies and scripts
```

## 🎨 UI Components

### Main Pages

- **Dashboard**: Overview with system status, agent summary, and recent activity
- **System Health**: Detailed health metrics and recommendations
- **Agents**: Agent monitoring and management
- **Metrics**: Performance charts and historical data
- **Security**: Security monitoring and threat detection
- **Settings**: System configuration

### Key Components

- `SystemStatusCard`: Shows overall system health score and status
- `AgentSummary`: Displays agent statistics and roles distribution
- `MetricsOverview`: Key performance metrics at a glance
- `RecentTasks`: List of recent task executions
- `PendingApprovals`: Human approval interface for critical operations

## 🚀 Development

### Run in development mode

```bash
cd dashboard
npm start
```

This will start the React development server on `http://localhost:3000` with hot reloading.

### Connect to backend

For development, you may need to configure proxy settings to connect to your Flask backend:

1. Create a `.env` file in the dashboard directory:
   ```
   REACT_APP_API_URL=http://localhost:5000/api
   ```

2. Or configure proxy in `package.json`:
   ```json
   "proxy": "http://localhost:5000"
   ```

## 📊 API Endpoints

The dashboard uses the following API endpoints:

- `GET /api/health/detailed` - Comprehensive health status
- `GET /api/health/enhanced` - Enhanced health with scoring
- `GET /api/agents` - Agent status and information
- `GET /api/metrics/stats` - Performance metrics
- `GET /api/task_history` - Recent task history
- `GET /api/catalyst/plans` - Pending approval plans
- `POST /api/approve` - Approve pending plans

## 🎯 Customization

### Theming

The dashboard uses Material-UI theming. You can customize colors in `src/index.js`:

```javascript
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',  // Change primary color
    },
    secondary: {
      main: '#f48fb1',  // Change secondary color
    },
  },
});
```

### Adding New Pages

1. Create a new page component in `src/pages/`
2. Add a route in `src/App.js`
3. Add a menu item in the sidebar

## 🔒 Security

The dashboard includes:

- JWT authentication support
- Error handling and user feedback
- Secure API communication
- Role-based access control (to be implemented)

## 📈 Future Enhancements

- Real-time WebSocket updates
- Advanced charting and visualization
- Alerting and notification system
- Multi-language support
- Customizable dashboards
- Export functionality for reports

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a pull request

## 📝 License

This dashboard is part of the Catalyst Vector Alpha platform and is licensed under the MIT License.