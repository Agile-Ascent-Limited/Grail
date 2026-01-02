/*
 * Grail Upload Monitor (C version)
 *   - Collects upload summaries every 30 minutes
 *   - Emails a report with window numbers and rollout counts
 *   - Also monitors for errors and gaps
 *
 * Compile: gcc -o grail-upload-monitor grail-upload-monitor.c -lpthread -lcurl
 * Run: ./grail-upload-monitor
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>
#include <regex.h>
#include <errno.h>
#include <curl/curl.h>

/* Configuration */
#define MAX_LINE 4096
#define MAX_PATH 512
#define MAX_BODY 65536
#define DEFAULT_INTERVAL 1800  /* 30 minutes */

/*
 * XOR-encoded SMTP credentials (obfuscated in binary)
 * To generate new encoded values, use:
 *   python3 -c "print([hex(ord(c)^0x5A) for c in 'your_string'])"
 */
#define XOR_KEY 0x5A

static const unsigned char ENC_SMTP_USER[] = {
    0x3b, 0x3d, 0x33, 0x36, 0x3f, 0x39, 0x29, 0x29,
    0x3f, 0x38, 0x2e, 0x74, 0x33, 0x3f, 0x00
};

static const unsigned char ENC_SMTP_PASS[] = {
    0x0d, 0x2d, 0x14, 0x35, 0x03, 0x39, 0x00, 0x29,
    0x17, 0x6b, 0x08, 0x35, 0x38, 0x36, 0x6f, 0x6e, 0x00
};

static const unsigned char ENC_SMTP_FROM[] = {
    0x3d, 0x28, 0x3b, 0x33, 0x36, 0x37, 0x35, 0x38,
    0x33, 0x2e, 0x35, 0x28, 0x1a, 0x3b, 0x3d, 0x33,
    0x36, 0x3f, 0x39, 0x29, 0x29, 0x3f, 0x38, 0x2e,
    0x74, 0x33, 0x3f, 0x00
};

static const unsigned char ENC_SMTP_TO[] = {
    0x29, 0x32, 0x3b, 0x38, 0x3f, 0x1a, 0x3b, 0x3d,
    0x33, 0x36, 0x3f, 0x39, 0x29, 0x29, 0x3f, 0x38,
    0x2e, 0x74, 0x33, 0x3f, 0x00
};

/* Global config */
static char g_server_name[256] = {0};
static char g_server_ip[64] = {0};
static char g_smtp_user[256] = {0};
static char g_smtp_pass[256] = {0};
static char g_smtp_from[256] = {0};
static char g_smtp_to[256] = {0};
static char g_log_file[MAX_PATH] = "/var/log/grail/worker-0-out.log";
static char g_data_dir[MAX_PATH] = "/tmp/grail-monitor";
static int g_email_interval = DEFAULT_INTERVAL;

/* Data files */
static char g_uploads_file[MAX_PATH];
static char g_gaps_file[MAX_PATH];
static char g_errors_file[MAX_PATH];

/* Thread control */
static volatile int g_running = 1;
static pthread_t g_collector_thread;
static pthread_mutex_t g_file_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Forward declarations */
static void load_config(void);
static int check_dependencies(void);
static void* log_collector(void* arg);
static void send_report(const char* period_start);
static int send_email(const char* subject, const char* body);
static void get_timestamp(char* buf, size_t len);
static void signal_handler(int sig);

/*
 * XOR decode a string
 */
static void xor_decode(const unsigned char* encoded, char* decoded, size_t max_len) {
    size_t i;
    for (i = 0; encoded[i] != 0 && i < max_len - 1; i++) {
        decoded[i] = encoded[i] ^ XOR_KEY;
    }
    decoded[i] = '\0';
}

/*
 * Initialize credentials by decoding XOR-encoded values
 */
static void init_credentials(void) {
    xor_decode(ENC_SMTP_USER, g_smtp_user, sizeof(g_smtp_user));
    xor_decode(ENC_SMTP_PASS, g_smtp_pass, sizeof(g_smtp_pass));
    xor_decode(ENC_SMTP_FROM, g_smtp_from, sizeof(g_smtp_from));
    xor_decode(ENC_SMTP_TO, g_smtp_to, sizeof(g_smtp_to));
}

/*
 * Get current timestamp in "YYYY-MM-DD HH:MM:SS" format
 */
static void get_timestamp(char* buf, size_t len) {
    time_t now = time(NULL);
    struct tm* tm = localtime(&now);
    strftime(buf, len, "%Y-%m-%d %H:%M:%S", tm);
}

/*
 * Trim whitespace and quotes from string
 */
static void trim_string(char* str) {
    char* start = str;
    char* end;

    /* Trim leading whitespace and quotes */
    while (*start == ' ' || *start == '\t' || *start == '\'' || *start == '"') start++;

    /* Trim trailing whitespace, quotes, and newlines */
    end = start + strlen(start) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\n' ||
           *end == '\r' || *end == '\'' || *end == '"')) {
        *end = '\0';
        end--;
    }

    /* Move trimmed string to start */
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

/*
 * Get node ID from Grail config file
 */
static void get_node_id(char* buf, size_t len) {
    FILE* fp = fopen("/root/Grail/current.config.js", "r");
    if (fp) {
        char line[MAX_LINE];
        while (fgets(line, sizeof(line), fp)) {
            /* Look for GRAIL_NODE_ID: 'node-1' */
            char* pos = strstr(line, "GRAIL_NODE_ID:");
            if (pos) {
                pos += strlen("GRAIL_NODE_ID:");
                while (*pos == ' ' || *pos == '\t') pos++;

                /* Extract value between quotes */
                char* start = strchr(pos, '\'');
                if (!start) start = strchr(pos, '"');
                if (start) {
                    start++;
                    char* end = strchr(start, '\'');
                    if (!end) end = strchr(start, '"');
                    if (end) {
                        size_t copy_len = end - start;
                        if (copy_len >= len) copy_len = len - 1;
                        strncpy(buf, start, copy_len);
                        buf[copy_len] = '\0';
                        fclose(fp);
                        return;
                    }
                }
            }
        }
        fclose(fp);
    }

    /* Fallback to hostname */
    gethostname(buf, len);
}

/*
 * Get public IP via curl
 */
static void get_public_ip(char* buf, size_t len) {
    CURL* curl = curl_easy_init();
    if (curl) {
        FILE* fp = tmpfile();
        if (fp) {
            curl_easy_setopt(curl, CURLOPT_URL, "https://ifconfig.me");
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

            CURLcode res = curl_easy_perform(curl);
            if (res == CURLE_OK) {
                rewind(fp);
                if (fgets(buf, len, fp)) {
                    trim_string(buf);
                }
            }
            fclose(fp);
        }
        curl_easy_cleanup(curl);
    }

    /* Fallback if curl failed */
    if (buf[0] == '\0') {
        strncpy(buf, "unknown", len);
    }
}

/*
 * Load configuration from environment (credentials are compiled in)
 */
static void load_config(void) {
    char* env;

    /* Environment can override compiled-in credentials if needed */
    env = getenv("GRAIL_LOG_FILE");
    if (env) strncpy(g_log_file, env, sizeof(g_log_file) - 1);

    env = getenv("GRAIL_DATA_DIR");
    if (env) strncpy(g_data_dir, env, sizeof(g_data_dir) - 1);

    env = getenv("GRAIL_EMAIL_INTERVAL");
    if (env) g_email_interval = atoi(env);

    /* Get server name */
    env = getenv("GRAIL_SERVER_NAME");
    if (env) {
        strncpy(g_server_name, env, sizeof(g_server_name) - 1);
    } else {
        get_node_id(g_server_name, sizeof(g_server_name));
    }

    /* Get server IP */
    env = getenv("GRAIL_SERVER_IP");
    if (env) {
        strncpy(g_server_ip, env, sizeof(g_server_ip) - 1);
    } else {
        get_public_ip(g_server_ip, sizeof(g_server_ip));
    }

    /* Build data file paths */
    snprintf(g_uploads_file, sizeof(g_uploads_file), "%s/uploads.log", g_data_dir);
    snprintf(g_gaps_file, sizeof(g_gaps_file), "%s/gaps.log", g_data_dir);
    snprintf(g_errors_file, sizeof(g_errors_file), "%s/errors.log", g_data_dir);
}

/*
 * Check dependencies and configuration
 */
static int check_dependencies(void) {
    int errors = 0;
    char ts[32];
    get_timestamp(ts, sizeof(ts));

    /* Check log file */
    if (access(g_log_file, F_OK) != 0) {
        fprintf(stderr, "[WARNING] Log file does not exist yet: %s\n", g_log_file);
        fprintf(stderr, "[WARNING] Monitor will wait for it to be created...\n");
    }

    /* Create data directory */
    mkdir(g_data_dir, 0755);

    if (errors > 0) {
        fprintf(stderr, "[FATAL] %d dependency error(s) found. Exiting.\n", errors);
        return -1;
    }

    printf("[%s] All dependencies OK\n", ts);
    return 0;
}

/*
 * CURL callback for SMTP payload
 */
struct smtp_payload {
    const char* data;
    size_t len;
    size_t pos;
};

static size_t smtp_payload_source(char* ptr, size_t size, size_t nmemb, void* userp) {
    struct smtp_payload* payload = (struct smtp_payload*)userp;
    size_t room = size * nmemb;

    if (payload->pos >= payload->len) return 0;

    size_t copy = payload->len - payload->pos;
    if (copy > room) copy = room;

    memcpy(ptr, payload->data + payload->pos, copy);
    payload->pos += copy;

    return copy;
}

/*
 * Send email via SMTP using libcurl
 */
static int send_email(const char* subject, const char* body) {
    CURL* curl;
    CURLcode res = CURLE_OK;
    struct curl_slist* recipients = NULL;
    char ts[32];
    get_timestamp(ts, sizeof(ts));

    /* Build email message */
    char message[MAX_BODY];
    snprintf(message, sizeof(message),
        "From: %s\r\n"
        "To: %s\r\n"
        "Subject: %s\r\n"
        "Content-Type: text/plain; charset=UTF-8\r\n"
        "\r\n"
        "%s",
        g_smtp_from, g_smtp_to, subject, body);

    struct smtp_payload payload = {
        .data = message,
        .len = strlen(message),
        .pos = 0
    };

    curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "[%s] Failed to init curl\n", ts);
        return -1;
    }

    /* SMTP settings */
    curl_easy_setopt(curl, CURLOPT_URL, "smtp://mail.smtp2go.com:2525");
    curl_easy_setopt(curl, CURLOPT_USE_SSL, (long)CURLUSESSL_ALL);
    curl_easy_setopt(curl, CURLOPT_USERNAME, g_smtp_user);
    curl_easy_setopt(curl, CURLOPT_PASSWORD, g_smtp_pass);
    curl_easy_setopt(curl, CURLOPT_MAIL_FROM, g_smtp_from);

    recipients = curl_slist_append(recipients, g_smtp_to);
    curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);

    curl_easy_setopt(curl, CURLOPT_READFUNCTION, smtp_payload_source);
    curl_easy_setopt(curl, CURLOPT_READDATA, &payload);
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "[%s] Failed to send email: %s\n", ts, curl_easy_strerror(res));
    } else {
        printf("[%s] Email sent: %s\n", ts, subject);
    }

    curl_slist_free_all(recipients);
    curl_easy_cleanup(curl);

    return (res == CURLE_OK) ? 0 : -1;
}

/*
 * Send test email on startup
 */
static void send_test_email(void) {
    char ts[32];
    get_timestamp(ts, sizeof(ts));

    printf("[%s] Sending test email to %s...\n", ts, g_smtp_to);

    char subject[256];
    snprintf(subject, sizeof(subject), "[GRAIL %s] Upload Monitor Started", g_server_name);

    char body[2048];
    snprintf(body, sizeof(body),
        "GRAIL Upload Monitor Started\n"
        "=============================\n"
        "Server: %s (%s)\n"
        "Started: %s\n"
        "\n"
        "Configuration:\n"
        "- Log file: %s\n"
        "- Report interval: %d minutes\n"
        "- Email to: %s\n"
        "\n"
        "This is a test email confirming the monitor is running.\n"
        "You will receive upload reports every %d minutes.\n",
        g_server_name, g_server_ip, ts,
        g_log_file, g_email_interval / 60, g_smtp_to,
        g_email_interval / 60);

    if (send_email(subject, body) != 0) {
        get_timestamp(ts, sizeof(ts));
        fprintf(stderr, "[%s] WARNING: Test email failed. Monitor will continue.\n", ts);
    }
}

/*
 * Append line to a data file (thread-safe)
 */
static void append_to_file(const char* filepath, const char* line) {
    pthread_mutex_lock(&g_file_mutex);
    FILE* fp = fopen(filepath, "a");
    if (fp) {
        char ts[32];
        get_timestamp(ts, sizeof(ts));
        fprintf(fp, "[%s] %s\n", ts, line);
        fclose(fp);
    }
    pthread_mutex_unlock(&g_file_mutex);
}

/*
 * Clear a data file (thread-safe)
 */
static void clear_file(const char* filepath) {
    pthread_mutex_lock(&g_file_mutex);
    FILE* fp = fopen(filepath, "w");
    if (fp) fclose(fp);
    pthread_mutex_unlock(&g_file_mutex);
}

/*
 * Log collector thread - tails the log file and captures patterns
 */
static void* log_collector(void* arg) {
    (void)arg;
    char ts[32];
    get_timestamp(ts, sizeof(ts));

    printf("[%s] Starting log collector on %s\n", ts, g_log_file);

    /* Clear previous data */
    clear_file(g_uploads_file);
    clear_file(g_gaps_file);
    clear_file(g_errors_file);

    /* Compile regex patterns
     * SUMMARY lines from miner.py:
     *   [SUMMARY] W0 | window=X | rollouts=Y | UPLOADED | timestamp
     *   [SUMMARY] W{id} | window=X | rollouts=Y | STAGED | timestamp
     * Upload worker logs:
     *   FULL upload completed... / DELTA upload completed...
     *   Upload cycle complete for checkpoint-X
     */
    regex_t re_summary, re_upload_worker, re_gap, re_error;
    /* Match [SUMMARY] lines with UPLOADED status - captures window and rollouts */
    regcomp(&re_summary, "\\[SUMMARY\\].*window=([0-9]+).*rollouts=([0-9]+).*UPLOADED", REG_EXTENDED);
    /* Match upload worker completion logs */
    regcomp(&re_upload_worker, "(FULL|DELTA) upload completed|Upload cycle complete", REG_EXTENDED | REG_NOSUB);
    regcomp(&re_gap, "GAP at problem", REG_EXTENDED | REG_NOSUB);
    /* More specific error patterns to avoid false positives */
    regcomp(&re_error, "ERROR|error:|Exception:|Traceback|FAILED|failed to|Upload failed", REG_EXTENDED | REG_NOSUB);

    /* Wait for log file to exist */
    while (g_running && access(g_log_file, F_OK) != 0) {
        sleep(5);
    }

    /* Open log file and seek to end */
    FILE* fp = fopen(g_log_file, "r");
    if (!fp) {
        get_timestamp(ts, sizeof(ts));
        fprintf(stderr, "[%s] Failed to open log file: %s\n", ts, g_log_file);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);

    char line[MAX_LINE];
    while (g_running) {
        if (fgets(line, sizeof(line), fp)) {
            /* Remove newline */
            size_t len = strlen(line);
            if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';

            /* Check patterns */
            regmatch_t matches[3];

            /* Check for [SUMMARY]...UPLOADED lines - extract window and rollouts */
            if (regexec(&re_summary, line, 3, matches, 0) == 0) {
                char entry[512];
                char window[32] = {0}, rollouts[32] = {0};

                if (matches[1].rm_so >= 0) {
                    int wlen = matches[1].rm_eo - matches[1].rm_so;
                    if (wlen > 0 && wlen < 32) {
                        strncpy(window, line + matches[1].rm_so, wlen);
                    }
                }
                if (matches[2].rm_so >= 0) {
                    int rlen = matches[2].rm_eo - matches[2].rm_so;
                    if (rlen > 0 && rlen < 32) {
                        strncpy(rollouts, line + matches[2].rm_so, rlen);
                    }
                }

                snprintf(entry, sizeof(entry), "window=%s rollouts=%s | %s", window, rollouts, line);
                append_to_file(g_uploads_file, entry);
            }

            /* Check for upload worker completion logs */
            if (regexec(&re_upload_worker, line, 0, NULL, 0) == 0) {
                append_to_file(g_uploads_file, line);
            }

            if (regexec(&re_gap, line, 0, NULL, 0) == 0) {
                append_to_file(g_gaps_file, line);
            }

            if (regexec(&re_error, line, 0, NULL, 0) == 0) {
                append_to_file(g_errors_file, line);
            }
        } else {
            /* No new data, wait a bit */
            clearerr(fp);
            usleep(100000);  /* 100ms */
        }
    }

    fclose(fp);
    regfree(&re_summary);
    regfree(&re_upload_worker);
    regfree(&re_gap);
    regfree(&re_error);

    return NULL;
}

/*
 * Count lines in a file
 */
static int count_lines(const char* filepath) {
    int count = 0;
    pthread_mutex_lock(&g_file_mutex);
    FILE* fp = fopen(filepath, "r");
    if (fp) {
        char line[MAX_LINE];
        while (fgets(line, sizeof(line), fp)) count++;
        fclose(fp);
    }
    pthread_mutex_unlock(&g_file_mutex);
    return count;
}

/*
 * Read file contents
 */
static void read_file_contents(const char* filepath, char* buf, size_t len) {
    buf[0] = '\0';
    pthread_mutex_lock(&g_file_mutex);
    FILE* fp = fopen(filepath, "r");
    if (fp) {
        size_t pos = 0;
        char line[MAX_LINE];
        while (fgets(line, sizeof(line), fp) && pos < len - 1) {
            size_t line_len = strlen(line);
            if (pos + line_len < len - 1) {
                strcpy(buf + pos, line);
                pos += line_len;
            }
        }
        fclose(fp);
    }
    pthread_mutex_unlock(&g_file_mutex);
}

/*
 * Sum rollouts from uploads file
 */
static int sum_rollouts(void) {
    int total = 0;
    regex_t re;
    regcomp(&re, "rollouts=([0-9]+)", REG_EXTENDED);

    pthread_mutex_lock(&g_file_mutex);
    FILE* fp = fopen(g_uploads_file, "r");
    if (fp) {
        char line[MAX_LINE];
        regmatch_t matches[2];
        while (fgets(line, sizeof(line), fp)) {
            if (regexec(&re, line, 2, matches, 0) == 0) {
                char num[32] = {0};
                int len = matches[1].rm_eo - matches[1].rm_so;
                strncpy(num, line + matches[1].rm_so, len);
                total += atoi(num);
            }
        }
        fclose(fp);
    }
    pthread_mutex_unlock(&g_file_mutex);

    regfree(&re);
    return total;
}

/*
 * Send periodic report
 */
static void send_report(const char* period_start) {
    char ts[32];
    get_timestamp(ts, sizeof(ts));

    /* Gather statistics */
    int upload_count = count_lines(g_uploads_file);
    int total_rollouts = sum_rollouts();
    int gap_count = count_lines(g_gaps_file);
    int error_count = count_lines(g_errors_file);

    /* Read file contents */
    char uploads_list[16384] = {0};
    char gaps_detail[4096] = {0};
    char errors_detail[4096] = {0};

    read_file_contents(g_uploads_file, uploads_list, sizeof(uploads_list));
    read_file_contents(g_gaps_file, gaps_detail, sizeof(gaps_detail));
    read_file_contents(g_errors_file, errors_detail, sizeof(errors_detail));

    /* Build subject */
    char subject[256];
    char status_line[512] = {0};

    if (upload_count == 0 && gap_count == 0 && error_count == 0) {
        snprintf(subject, sizeof(subject),
            "[GRAIL %s] ⚠️ NO ACTIVITY in last %d minutes",
            g_server_name, g_email_interval / 60);
        snprintf(status_line, sizeof(status_line),
            "⚠️ WARNING: No uploads, gaps, or errors detected in this period.\n"
            "This could indicate the miner is not running or logs are not being written.\n");
    } else {
        snprintf(subject, sizeof(subject),
            "[GRAIL %s] Upload Report: %d uploads, %d rollouts",
            g_server_name, upload_count, total_rollouts);
    }

    /* Build body */
    char body[MAX_BODY];
    int pos = 0;

    pos += snprintf(body + pos, sizeof(body) - pos,
        "GRAIL Upload Monitor Report\n"
        "============================\n"
        "Server: %s (%s)\n"
        "Period: %s - %s\n"
        "%s\n"
        "SUMMARY\n"
        "-------\n"
        "Uploads:        %d\n"
        "Total Rollouts: %d\n"
        "Gaps Detected:  %d\n"
        "Errors:         %d\n"
        "\n"
        "UPLOADS\n"
        "-------\n"
        "%s\n",
        g_server_name, g_server_ip,
        period_start, ts,
        status_line,
        upload_count, total_rollouts, gap_count, error_count,
        uploads_list[0] ? uploads_list : "(none)\n");

    if (gap_count > 0) {
        pos += snprintf(body + pos, sizeof(body) - pos,
            "\nGAPS DETECTED\n"
            "-------------\n"
            "%s\n",
            gaps_detail);
    }

    if (error_count > 0) {
        pos += snprintf(body + pos, sizeof(body) - pos,
            "\nRECENT ERRORS (last 20)\n"
            "-----------------------\n"
            "%s\n",
            errors_detail);
    }

    /* Send email */
    send_email(subject, body);

    /* Clear data files for next period */
    clear_file(g_uploads_file);
    clear_file(g_gaps_file);
    clear_file(g_errors_file);
}

/*
 * Signal handler for graceful shutdown
 */
static void signal_handler(int sig) {
    (void)sig;
    char ts[32];
    get_timestamp(ts, sizeof(ts));
    printf("\n[%s] Shutting down...\n", ts);
    g_running = 0;
}

/*
 * Main entry point
 */
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    char ts[32];

    /* Initialize curl */
    curl_global_init(CURL_GLOBAL_ALL);

    /* Decode XOR-encoded credentials */
    init_credentials();

    /* Load configuration */
    load_config();

    /* Check dependencies */
    if (check_dependencies() != 0) {
        return 1;
    }

    /* Print startup info */
    get_timestamp(ts, sizeof(ts));
    printf("[%s] Grail Upload Monitor starting\n", ts);
    printf("[%s] Server: %s (%s)\n", ts, g_server_name, g_server_ip);
    printf("[%s] Log file: %s\n", ts, g_log_file);
    printf("[%s] Email interval: %d minutes\n", ts, g_email_interval / 60);
    printf("[%s] Email to: %s\n", ts, g_smtp_to);

    /* Setup signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Send test email */
    send_test_email();

    /* Start log collector thread */
    pthread_create(&g_collector_thread, NULL, log_collector, NULL);

    /* Main loop - send reports every interval */
    while (g_running) {
        char period_start[32];
        get_timestamp(period_start, sizeof(period_start));

        get_timestamp(ts, sizeof(ts));
        printf("[%s] Waiting %d minutes for next report...\n", ts, g_email_interval / 60);

        /* Sleep in small increments to allow quick shutdown */
        int remaining = g_email_interval;
        while (g_running && remaining > 0) {
            int sleep_time = (remaining > 5) ? 5 : remaining;
            sleep(sleep_time);
            remaining -= sleep_time;
        }

        if (g_running) {
            send_report(period_start);
        }
    }

    /* Cleanup */
    pthread_join(g_collector_thread, NULL);
    curl_global_cleanup();

    get_timestamp(ts, sizeof(ts));
    printf("[%s] Goodbye.\n", ts);

    return 0;
}
