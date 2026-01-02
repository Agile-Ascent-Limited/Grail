#!/usr/bin/env bash
#
# Grail Upload Monitor
#   - Collects upload summaries every 30 minutes
#   - Emails a report with window numbers and rollout counts
#   - Also monitors for errors and gaps
# ------------------------------------------------------------------------------

#### Configuration (set these or use environment variables) ####
SERVER_NAME="${GRAIL_SERVER_NAME:-$(hostname)}"
SERVER_IP="${GRAIL_SERVER_IP:-$(hostname -I | awk '{print $1}')}"
EMAIL_INTERVAL="${GRAIL_EMAIL_INTERVAL:-1800}"  # 30 minutes in seconds

# SMTP settings (smtp2go)
SMTP_USER="${GRAIL_SMTP_USER:-agileascent.ie}"
SMTP_PASS="${GRAIL_SMTP_PASS:-WwNoYcZsM1Ronl54}"
SMTP_FROM="${GRAIL_SMTP_FROM:-grailmonitor@agileascent.ie}"
SMTP_TO="${GRAIL_SMTP_TO:-shane@agileascent.ie}"

########## paths ##########
LOG_FILE="${GRAIL_LOG_FILE:-/var/log/grail/worker-0-out.log}"
DATA_DIR="${GRAIL_DATA_DIR:-/tmp/grail-monitor}"
UPLOADS_FILE="$DATA_DIR/uploads.log"
GAPS_FILE="$DATA_DIR/gaps.log"
ERRORS_FILE="$DATA_DIR/errors.log"

# Create data directory
mkdir -p "$DATA_DIR"

################################################################################
# Dependency checks
################################################################################
check_dependencies() {
  local errors=0

  # Check Python 3
  if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 is not installed. Install with: apt install python3"
    errors=$((errors + 1))
  else
    # Check Python SMTP modules (standard library, but verify)
    if ! python3 -c "import smtplib, email.mime.text, email.mime.multipart" 2>/dev/null; then
      echo "[ERROR] Python email modules not available"
      errors=$((errors + 1))
    fi
  fi

  # Check tail command
  if ! command -v tail &>/dev/null; then
    echo "[ERROR] tail command not found"
    errors=$((errors + 1))
  fi

  # Check log file exists
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "[WARNING] Log file does not exist yet: $LOG_FILE"
    echo "[WARNING] Monitor will wait for it to be created..."
  fi

  # Check SMTP connectivity (optional, just a warning)
  if command -v nc &>/dev/null; then
    if ! nc -z -w 5 mail.smtp2go.com 2525 2>/dev/null; then
      echo "[WARNING] Cannot reach SMTP server mail.smtp2go.com:2525"
      echo "[WARNING] Emails may fail to send. Check network/firewall."
    fi
  fi

  if [[ $errors -gt 0 ]]; then
    echo "[FATAL] $errors dependency error(s) found. Exiting."
    exit 1
  fi

  echo "[$(date +'%F %T')] All dependencies OK"
}

# Run dependency checks
check_dependencies

################################################################################
# Helper: e-mail sender
################################################################################
send_test_email() {
  echo "[$(date +'%F %T')] Sending test email to $SMTP_TO..."

  python3 - "$SERVER_NAME" "$SERVER_IP" "$LOG_FILE" "$EMAIL_INTERVAL" "$SMTP_USER" "$SMTP_PASS" "$SMTP_FROM" "$SMTP_TO" <<'PY'
import sys, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

server_name, server_ip, log_file, interval, smtp_user, smtp_pass, sender, receiver = sys.argv[1:]

msg = MIMEMultipart()
msg['Subject'] = f"[GRAIL {server_name}] Upload Monitor Started"
msg['From']    = sender
msg['To']      = receiver

body = f"""GRAIL Upload Monitor Started
=============================
Server: {server_name} ({server_ip})
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
- Log file: {log_file}
- Report interval: {int(interval)//60} minutes
- Email to: {receiver}

This is a test email confirming the monitor is running.
You will receive upload reports every {int(interval)//60} minutes.
"""

msg.attach(MIMEText(body, 'plain'))

try:
    with smtplib.SMTP('mail.smtp2go.com', 2525) as s:
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.sendmail(sender, receiver, msg.as_string())
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test email sent successfully")
except Exception as e:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Test email failed: {e}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitor will continue, but emails may not work")
PY
}

################################################################################
# Helper: e-mail sender (for reports)
################################################################################
send_email() {
  local subject="$1"
  local body="$2"

  python3 - "$subject" "$body" "$SMTP_USER" "$SMTP_PASS" "$SMTP_FROM" "$SMTP_TO" <<'PY'
import sys, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

subject, body, smtp_user, smtp_pass, sender, receiver = sys.argv[1:]

msg = MIMEMultipart()
msg['Subject'] = subject
msg['From']    = sender
msg['To']      = receiver
msg.attach(MIMEText(body, 'plain'))

try:
    with smtplib.SMTP('mail.smtp2go.com', 2525) as s:
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.sendmail(sender, receiver, msg.as_string())
    print(f"Email sent: {subject}")
except Exception as e:
    print(f"Failed to send email: {e}")
PY
}

################################################################################
# Log collector - runs in background, appends to data files
################################################################################
collect_logs() {
  echo "[$(date +'%F %T')] Starting log collector on $LOG_FILE"

  # Clear previous data
  > "$UPLOADS_FILE"
  > "$GAPS_FILE"
  > "$ERRORS_FILE"

  tail -n0 -F "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    # Capture upload summaries: [SUMMARY] W0 | window=X | rollouts=Y | UPLOADED | HH:MM:SS
    if [[ "$line" =~ \[SUMMARY\].*UPLOADED ]]; then
      echo "[$(date +'%F %T')] $line" >> "$UPLOADS_FILE"
    fi

    # Capture successful uploads with rollout counts
    if [[ "$line" =~ Successfully\ uploaded\ window\ ([0-9]+)\ with\ ([0-9]+)\ aggregated\ rollouts ]]; then
      window="${BASH_REMATCH[1]}"
      rollouts="${BASH_REMATCH[2]}"
      echo "[$(date +'%F %T')] window=$window rollouts=$rollouts" >> "$UPLOADS_FILE"
    fi

    # Capture gap warnings
    if [[ "$line" =~ GAP\ at\ problem ]]; then
      echo "[$(date +'%F %T')] $line" >> "$GAPS_FILE"
    fi

    # Capture errors
    if [[ "$line" =~ ERROR|error:|Exception|Traceback ]]; then
      echo "[$(date +'%F %T')] $line" >> "$ERRORS_FILE"
    fi
  done
}

################################################################################
# Report generator - runs every EMAIL_INTERVAL
################################################################################
send_report() {
  local period_start="$1"
  local period_end
  period_end=$(date +'%F %T')

  # Count uploads and sum rollouts
  local upload_count=0
  local total_rollouts=0
  local windows_list=""

  if [[ -f "$UPLOADS_FILE" && -s "$UPLOADS_FILE" ]]; then
    upload_count=$(wc -l < "$UPLOADS_FILE")

    # Extract rollout counts and sum them
    while IFS= read -r line; do
      if [[ "$line" =~ rollouts=([0-9]+) ]]; then
        rollouts="${BASH_REMATCH[1]}"
        total_rollouts=$((total_rollouts + rollouts))
      fi
      # Also try the SUMMARY format
      if [[ "$line" =~ \|\ rollouts=([0-9]+)\ \| ]]; then
        rollouts="${BASH_REMATCH[1]}"
        total_rollouts=$((total_rollouts + rollouts))
      fi
    done < "$UPLOADS_FILE"

    windows_list=$(cat "$UPLOADS_FILE")
  fi

  # Count gaps
  local gap_count=0
  local gaps_detail=""
  if [[ -f "$GAPS_FILE" && -s "$GAPS_FILE" ]]; then
    gap_count=$(wc -l < "$GAPS_FILE")
    gaps_detail=$(cat "$GAPS_FILE")
  fi

  # Count errors (last 20)
  local error_count=0
  local errors_detail=""
  if [[ -f "$ERRORS_FILE" && -s "$ERRORS_FILE" ]]; then
    error_count=$(wc -l < "$ERRORS_FILE")
    errors_detail=$(tail -20 "$ERRORS_FILE")
  fi

  # Build report
  local subject="[GRAIL $SERVER_NAME] Upload Report: $upload_count uploads, $total_rollouts rollouts"

  local body
  body=$(cat <<EOF
GRAIL Upload Monitor Report
============================
Server: $SERVER_NAME ($SERVER_IP)
Period: $period_start - $period_end

SUMMARY
-------
Uploads:       $upload_count
Total Rollouts: $total_rollouts
Gaps Detected: $gap_count
Errors:        $error_count

UPLOADS
-------
$windows_list

EOF
)

  if [[ $gap_count -gt 0 ]]; then
    body+=$(cat <<EOF

GAPS DETECTED
-------------
$gaps_detail

EOF
)
  fi

  if [[ $error_count -gt 0 ]]; then
    body+=$(cat <<EOF

RECENT ERRORS (last 20)
-----------------------
$errors_detail

EOF
)
  fi

  # Send email
  send_email "$subject" "$body"

  # Clear data files for next period
  > "$UPLOADS_FILE"
  > "$GAPS_FILE"
  > "$ERRORS_FILE"
}

################################################################################
# Main loop
################################################################################
echo "[$(date +'%F %T')] Grail Upload Monitor starting"
echo "[$(date +'%F %T')] Server: $SERVER_NAME ($SERVER_IP)"
echo "[$(date +'%F %T')] Log file: $LOG_FILE"
echo "[$(date +'%F %T')] Email interval: $((EMAIL_INTERVAL/60)) minutes"
echo "[$(date +'%F %T')] Email to: $SMTP_TO"

# Send test email on startup
send_test_email

# Start log collector in background
collect_logs &
COLLECTOR_PID=$!

# Cleanup on exit
trap 'kill $COLLECTOR_PID 2>/dev/null; exit 0' EXIT INT TERM

# Send reports every EMAIL_INTERVAL
while true; do
  period_start=$(date +'%F %T')
  echo "[$(date +'%F %T')] Waiting $((EMAIL_INTERVAL/60)) minutes for next report..."
  sleep "$EMAIL_INTERVAL"
  send_report "$period_start"
done
