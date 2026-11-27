# v0.4 서버 동작 시험 (가짜 학습기·가짜 sd-cli로. GPU·모델 불필요)
# 사용: python3 tests/test_server.py build/sdxl_lora_server
# 임시 폴더에 가짜 프로그램과 작업 폴더를 만들고 포트 8091에서 서버를 띄워 56항목을 확인한다. 끝나면 임시 폴더를 지운다.
import json, os, shutil, signal, subprocess, sys, tempfile, time, urllib.request, urllib.error

SERVER = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'build/sdxl_lora_server')
T = tempfile.mkdtemp(prefix='sdxl_srvtest_')
ROOT = os.path.join(T, 'root')
FAKE_LORA = r'''#!/bin/bash
# 가짜 학습기: --out, --steps만 봄. steps=999면 SIGTERM 무시 + 손자 프로세스(sleep)를 만들어 그룹 종료 시험
out=""; steps=5; while [ $# -gt 0 ]; do case $1 in --out) out=$2; shift;; --steps) steps=$2; shift;; esac; shift; done
echo "args ok, cwd=$(pwd)"
if [ "$steps" = 999 ]; then trap '' TERM; sleep 1000 & echo "grandchild $!" ; fi
trap 'echo "got TERM"; exit 143' TERM 2>/dev/null; [ "$steps" = 999 ] && trap '' TERM
for i in $(seq 1 $steps); do echo "step $i  t=100  loss 0.1  ema 0.1  0.30s  GTT 0 MiB  [x]  S=32768"; sleep 0.3; done
echo lora > "$out"; echo "저장: $out"
'''
FAKE_SDCLI = r'''#!/bin/bash
# 가짜 sd-cli: -M dgl --script <f> 이면 스크립트 속 save 경로에 파일 생성, 아니면 -o 에 그림 생성
script=""; out=""; while [ $# -gt 0 ]; do case $1 in --script) script=$2; shift;; -o) out=$2; shift;; esac; shift; done
if [ -n "$script" ]; then grep -o '::save("[^"]*"' "$script" | sed 's/::save("//;s/"$//' | while read f; do echo x > "$f"; done; echo "dgl done"; exit 0; fi
echo "[I] sampling started: method=euler_a, steps=4"
for i in 1 2 3 4; do printf '  |==>   |  %d/4 -   0.20s/it - ETA 00:01\r' $i; sleep 0.2; done; echo
echo png > "$out"; echo "save result"
'''
os.makedirs(os.path.join(T, 'bin'))
for name, body in (('fake_lora', FAKE_LORA), ('fake_sdcli', FAKE_SDCLI)):
    with open(os.path.join(T, 'bin', name), 'w') as f:
        f.write(body)
    os.chmod(os.path.join(T, 'bin', name), 0o755)
os.makedirs(os.path.join(ROOT, 'dataset/dogs'))
os.makedirs(os.path.join(ROOT, 'cache/ready'))
for fn, body in (('a.jpg', 'photo'), ('b.png', 'photo'), ('a.txt', 'a cute dog')):
    open(os.path.join(ROOT, 'dataset/dogs', fn), 'w').write(body + '\n')
for n in ('x', 'y'):
    open(os.path.join(ROOT, 'cache/ready', n + '.latent.npy'), 'w').write('1\n')
URL = 'http://127.0.0.1:8091'
srv = None
results = []


def start(extra=()):
    global srv
    srv = subprocess.Popen([SERVER, '--root', ROOT, '--cli', os.path.join(T, 'bin/fake_lora'),
                            '--sd-cli', os.path.join(T, 'bin/fake_sdcli'), '--guard', 'off', '--port', '8091', *extra],
                           stdout=open(os.path.join(T, 'server.out'), 'a'), stderr=subprocess.STDOUT)
    for _ in range(150):
        try:
            urllib.request.urlopen(URL + '/', timeout=1)
            return
        except Exception:
            time.sleep(0.1)
    raise SystemExit('서버가 안 뜸')


def call(path, body=None, headers=None):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(), headers=headers or {})
    if body is not None:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw[:1] in (b'{', b'[') else raw.decode())
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw[:1] == b'{' else raw.decode())


def jobs():
    return {j['id']: j for j in call('/api/jobs')[1]['jobs']}


def wait(cond, timeout=30, what=''):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return time.time() - t0
        time.sleep(0.2)
    raise AssertionError('시간 초과: ' + what)


def check(name, ok, detail=''):
    results.append((name, ok))
    print(('PASS ' if ok else 'FAIL ') + name + (' — ' + detail if detail else ''))


def alive(pid):
    try:
        os.kill(pid, 0)
        st = open(f'/proc/{pid}/stat').read().split()[2]
        return st != 'Z'
    except (OSError, FileNotFoundError):
        return False


def grandchild(jid):
    for ln in open(os.path.join(ROOT, 'jobs', str(jid), 'log.txt')):
        if ln.startswith('grandchild '):
            return int(ln.split()[1])
    return None


start()
c, cfg = call('/api/config')
check('config', c == 200 and cfg['sd_cli'] and cfg['cache'] and not cfg['token'], json.dumps(cfg))

# 1) 파이프라인: 캐시 → 학습 → 생성 2장 (뒤 작업은 앞 작업이 만들 파일을 씀)
ids = [call('/api/jobs', {'type': 'cache', 'src': 'dataset/dogs', 'size': 512, 'caption': 'sks dog', 'prefix': 'p, '})[1]['id'],
       call('/api/jobs', {'type': 'train', 'data': 'cache/dogs_512', 'steps': 5})[1]['id'],
       call('/api/jobs', {'type': 'generate', 'prompt': 'dog', 'lora': 'out/dogs_512.safetensors', 'width': 512, 'height': 512})[1]['id'],
       call('/api/jobs', {'type': 'generate', 'prompt': '한글 프롬프트 "따옴표"', 'width': 512, 'height': 512, 'seed': 7})[1]['id']]
wait(lambda: all(jobs()[i]['state'] in ('done', 'failed') for i in ids), 60, '파이프라인')
J = jobs()
check('파이프라인 4개 모두 done', all(J[i]['state'] == 'done' for i in ids), str([(i, J[i]['state'], J[i]['error']) for i in ids]))
order = sorted(ids, key=lambda i: J[i]['started'])
check('차례 실행 (시작 순서 = 넣은 순서)', order == ids)
check('겹침 없음 (앞 작업 끝난 뒤 다음 시작)', all(J[ids[k]]['finished'] <= J[ids[k + 1]]['started'] for k in range(3)))
cap = open(os.path.join(ROOT, 'cache/dogs_512/captions.txt'), encoding='utf-8').read()
check('캐시: 캡션 .txt 우선 + prefix', 'p, a cute dog' in cap and 'p, sks dog' in cap, cap.strip())
dgl = open(os.path.join(ROOT, 'cache/dogs_512/cache.dgl')).read()
check('캐시: DGL 스크립트 (make_cache.py와 같은 형식)', 'fn prep(path, size)' in dgl and 'vae_to_unet_latents' in dgl and dgl.count('encode_prompt(text,') == 2)
check('캐시 결과 파일', all(os.path.exists(os.path.join(ROOT, 'cache/dogs_512', f)) for f in ['a.latent.npy', 'b.vector.npy', 'a.check.png']))
d = call(f'/api/jobs/{ids[2]}')[1]
check('생성: LoRA 태그가 프롬프트에, 출력 그림', any('<lora:dogs_512:1>' in l for l in d['log_tail']) and d['outputs'][0]['kind'] == 'image', str(d['outputs']))
check('생성 진행률 파싱', d.get('progress', {}).get('step') == 4, str(d.get('progress')))
d = call(f'/api/jobs/{ids[1]}')[1]
check('학습 진행률 파싱 + LoRA 결과', d.get('progress', {}).get('step') == 5 and any(o['kind'] == 'lora' for o in d['outputs']))
c, body = call('/files/' + d['outputs'][0]['path'])
check('결과 내려받기 (/files)', c == 200 and 'lora' in body)

# 2) 순서 바꾸기 + 일시정지
call('/api/queue/pause', {})
a, b, cc = [call('/api/jobs', {'type': 'train', 'data': 'cache/ready', 'steps': 1, 'out': f'out/m{k}.safetensors'})[1]['id'] for k in range(3)]
call(f'/api/jobs/{cc}/move', {'dir': 'top'})
call(f'/api/jobs/{a}/move', {'dir': 'down'})
time.sleep(1)
check('일시정지 중 시작 안 함', all(jobs()[i]['state'] == 'queued' for i in (a, b, cc)))
call('/api/queue/resume', {})
wait(lambda: all(jobs()[i]['state'] == 'done' for i in (a, b, cc)), 30, '순서')
J = jobs()
check('순서 바꾸기 (top, down) 반영', sorted((a, b, cc), key=lambda i: J[i]['started']) == [cc, b, a])

# 3) 대기 취소 / 실행 중 정상 중지 / SIGTERM 무시 → 10초 후 강제 / 즉시 강제
call('/api/queue/pause', {})
q = call('/api/jobs', {'type': 'train', 'data': 'cache/ready', 'steps': 3})[1]['id']
call(f'/api/jobs/{q}/cancel', {})
check('대기 작업 취소', jobs()[q]['state'] == 'cancelled')
call('/api/queue/resume', {})
r = call('/api/jobs', {'type': 'train', 'data': 'cache/ready', 'steps': 100})[1]['id']
wait(lambda: jobs()[r]['state'] == 'running', 10, 'r 시작')
time.sleep(0.5)
call(f'/api/jobs/{r}/cancel', {})
dt = wait(lambda: jobs()[r]['state'] == 'stopped', 15, '정상 중지')
check('실행 중 취소 (SIGTERM으로 바로 끝남)', dt < 3 and 'got TERM' in open(os.path.join(ROOT, f'jobs/{r}/log.txt')).read(), f'{dt:.1f}s')

h = call('/api/jobs', {'type': 'train', 'data': 'cache/ready', 'steps': 999})[1]['id']
wait(lambda: jobs()[h]['state'] == 'running' and grandchild(h), 10, 'h 시작')
gc = grandchild(h)
call(f'/api/jobs/{h}/cancel', {})
dt = wait(lambda: jobs()[h]['state'] == 'stopped', 25, 'TERM 무시 작업')
time.sleep(0.3)
check('SIGTERM 무시하는 작업 → 10초 뒤 자동 SIGKILL', 9 < dt < 14, f'{dt:.1f}s')
check('손자 프로세스까지 종료 (그룹 kill)', not alive(gc), f'pid {gc}')

k = call('/api/jobs', {'type': 'train', 'data': 'cache/ready', 'steps': 999})[1]['id']
wait(lambda: jobs()[k]['state'] == 'running' and grandchild(k), 10, 'k 시작')
gc = grandchild(k)
call(f'/api/jobs/{k}/kill', {})
dt = wait(lambda: jobs()[k]['state'] == 'stopped', 5, '즉시 kill')
time.sleep(0.3)
check('즉시 강제 종료 (kill)', dt < 2 and not alive(gc), f'{dt:.1f}s, 손자 살아있음={alive(gc)}')

# 4) 서버 재시작: 돌던 작업은 interrupted, 남은 프로세스 정리, 대기 작업은 이어서 실행
s1 = call('/api/jobs', {'type': 'train', 'data': 'cache/ready', 'steps': 999})[1]['id']
s2 = call('/api/jobs', {'type': 'train', 'data': 'cache/ready', 'steps': 2, 'out': 'out/after.safetensors'})[1]['id']
wait(lambda: jobs()[s1]['state'] == 'running' and grandchild(s1), 10, 's1 시작')
gc = grandchild(s1)
srv.send_signal(signal.SIGKILL)
srv.wait()
time.sleep(0.5)
check('서버가 죽어도 작업 프로세스는 남아 있음 (재시작 시 정리 대상)', alive(gc))
start()
J = jobs()
check('재시작 후 돌던 작업 = interrupted', J[s1]['state'] == 'interrupted', J[s1]['state'])
check('남은 작업 프로세스(손자 포함) 정리', not alive(gc))
wait(lambda: jobs()[s2]['state'] == 'done', 20, 's2')
check('대기 작업은 재시작 후 이어서 실행', True)

# 5) 경로 검사
c, e = call('/api/jobs', {'type': 'train', 'data': '../../etc'})
check('root 밖 경로 거부 (train data)', c == 400, str(e))
c, e = call('/api/jobs', {'type': 'generate', 'prompt': 'x', 'lora': '/etc/passwd'})
check('root 밖 경로 거부 (lora 절대경로)', c == 400, str(e))
c, e = call('/api/jobs', {'type': 'generate', 'prompt': 'x <lora:../../evil:1>'})
q2 = e.get('id') if c == 200 else None
if q2:
    wait(lambda: jobs()[q2]['state'] in ('done', 'failed'), 10)
check('프롬프트 안의 <lora:> 직접 지정 거부', c == 400 or jobs()[q2]['state'] == 'failed', str(e))
c, _ = call('/files/..%2f..%2fetc/passwd')
check('/files root 밖 거부', c == 404, str(c))
c, e = call('/api/jobs', {'type': 'nope'})
check('잘못된 type 거부', c == 400)
c, e = call('/api/jobs', {'type': 'generate', 'prompt': 'x', 'width': 500})
q3 = e.get('id')
wait(lambda: jobs()[q3]['state'] in ('done', 'failed'), 10)
check('잘못된 크기 → 작업 실패 + 이유 기록', jobs()[q3]['state'] == 'failed' and '64' in jobs()[q3]['error'], jobs()[q3]['error'])

# 6) 선택 삭제 (파일 포함). 캐시 출력 위치를 일부러 원본 사진 폴더로 준 작업도 지워 봄 → 원본은 남아야 함
danger = call('/api/jobs', {'type': 'cache', 'src': 'dataset/dogs', 'dst': 'dataset/dogs', 'size': 512})[1]['id']
wait(lambda: jobs()[danger]['state'] == 'done', 20, 'danger cache')
c, e = call('/api/jobs/remove-many', {'ids': [ids[0], ids[1], ids[2], danger, s2], 'delete_files': True})
check('선택 삭제 응답', c == 200 and e['removed'] == 5, str(e))
left = sorted(os.listdir(os.path.join(ROOT, 'dataset/dogs')))
check('원본 사진·캡션은 보존 (캐시 산출물만 삭제)', left == ['a.jpg', 'a.txt', 'b.png'], str(left))
check('캐시 폴더 삭제 (비었으므로)', not os.path.exists(os.path.join(ROOT, 'cache/dogs_512')))
check('학습 LoRA 삭제', not os.path.exists(os.path.join(ROOT, 'out/dogs_512.safetensors')) and not os.path.exists(os.path.join(ROOT, 'out/after.safetensors')))
check('생성 작업 폴더 삭제', not os.path.exists(os.path.join(ROOT, f'jobs/{ids[2]}')))
check('다른 작업 결과는 그대로', os.path.exists(os.path.join(ROOT, 'out/m0.safetensors')) and os.path.exists(os.path.join(ROOT, f'jobs/{ids[3]}/image.png')))
c, e = call('/api/jobs/remove-many', {'ids': [ids[3]]})
check('파일 없이 목록에서만 삭제', e['removed'] == 1 and os.path.exists(os.path.join(ROOT, f'jobs/{ids[3]}/image.png')))
call('/api/queue/pause', {})
qq = call('/api/jobs', {'type': 'train', 'data': 'cache/ready', 'steps': 1})[1]['id']
c, e = call('/api/jobs/remove-many', {'ids': [qq], 'delete_files': True})
check('대기 중 작업은 삭제 안 함', e['removed'] == 0 and qq in e['skipped_active'])
c, e = call('/api/queue/cancel-all', {})
check('대기 전부 취소', e['count'] >= 1 and jobs()[qq]['state'] == 'cancelled')
call('/api/queue/resume', {})

# 7) v0.3 호환 API
c, e = call('/api/train', {'data': 'cache/ready', 'steps': 50})
cid = e['id']
wait(lambda: jobs()[cid]['state'] == 'running', 10)
time.sleep(1)
c, st = call('/api/status')
check('호환 /api/status (step, sec_per_step)', st['state'] == 'running' and 'step' in st and 'sec_per_step' in st, json.dumps({k: st.get(k) for k in ('state', 'step', 'steps')}))
c, e = call('/api/stop', {})
wait(lambda: jobs()[cid]['state'] == 'stopped', 10)
check('호환 /api/stop', c == 200)
c, log = call('/api/log')
check('호환 /api/log', 'step 1' in log)

# 8) 토큰
srv.terminate(); srv.wait()
start(['--token', 's3cret'])
check('토큰: 없으면 401', call('/api/jobs')[0] == 401)
check('토큰: 헤더 OK', call('/api/jobs', headers={'Authorization': 'Bearer s3cret'})[0] == 200)
check('토큰: 쿠키 OK', call('/api/jobs', headers={'Cookie': 'a=1; sdxl_token=s3cret'})[0] == 200)
check('토큰: 틀리면 401', call('/api/jobs', headers={'Authorization': 'Bearer nope'})[0] == 401)
check('토큰: /files도 보호', call('/files/out/m0.safetensors')[0] == 401)
check('토큰: 첫 화면(/)은 열림', call('/')[0] == 200)
srv.terminate(); srv.wait()

# 9) 웹 화면 끄기 / 직접 만든 프론트 / CORS
start(['--ui', 'off'])
c, body = call('/')
check('--ui off: / 는 화면 대신 API 안내 JSON', c == 200 and isinstance(body, dict) and body.get('ui') == 'off', str(body)[:80])
check('--ui off: API는 그대로', call('/api/jobs')[0] == 200)
srv.terminate(); srv.wait()
ui = os.path.join(T, 'myui')
os.makedirs(os.path.join(ui, 'api'))
open(os.path.join(ui, 'index.html'), 'w').write('<!doctype html><title>my frontend</title>')
open(os.path.join(ui, 'app.js'), 'w').write('console.log(1)')
open(os.path.join(ui, 'api', 'jobs'), 'w').write('should not be served without token')
start(['--ui', ui, '--token', 'tok2', '--cors', 'http://localhost:5173'])
c, body = call('/')
check('--ui <폴더>: / 에 직접 만든 index.html', c == 200 and 'my frontend' in body, str(body)[:60])
check('--ui <폴더>: 프론트 파일은 토큰 없이', call('/app.js')[0] == 200)
check('--ui <폴더>: API는 여전히 토큰 필요 (폴더의 api/ 파일로 우회 불가)', call('/api/jobs')[0] == 401)
check('--ui <폴더>: 토큰 주면 API 정상', call('/api/jobs', headers={'Authorization': 'Bearer tok2'})[0] == 200)
req = urllib.request.Request(URL + '/api/jobs', method='OPTIONS', headers={'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'GET'})
with urllib.request.urlopen(req, timeout=5) as r:
    pre = (r.status, r.headers.get('Access-Control-Allow-Origin'), r.headers.get('Access-Control-Allow-Headers'))
check('--cors: 사전 요청(OPTIONS) 204 + 허용 헤더', pre[0] == 204 and pre[1] == 'http://localhost:5173' and 'Authorization' in (pre[2] or ''), str(pre))
req = urllib.request.Request(URL + '/api/jobs', headers={'Authorization': 'Bearer tok2', 'Origin': 'http://localhost:5173'})
with urllib.request.urlopen(req, timeout=5) as r:
    acao = r.headers.get('Access-Control-Allow-Origin')
check('--cors: API 응답에 허용 헤더', acao == 'http://localhost:5173', str(acao))
srv.terminate(); srv.wait()
bad_cors = subprocess.run([SERVER, '--root', ROOT, '--cli', os.path.join(T, 'bin/fake_lora'), '--cors', 'evil.com', '--port', '8091'], capture_output=True, timeout=10)
check('--cors 잘못된 값 거부', bad_cors.returncode == 2)
bad_ui = subprocess.run([SERVER, '--root', ROOT, '--cli', os.path.join(T, 'bin/fake_lora'), '--ui', os.path.join(T, 'nope'), '--port', '8091'], capture_output=True, timeout=10)
check('--ui 없는 폴더 거부', bad_ui.returncode == 2)

bad = [n for n, ok in results if not ok]
shutil.rmtree(T, ignore_errors=True)
print(f'\n{len(results) - len(bad)}/{len(results)} PASS' + (' — 실패: ' + ', '.join(bad) if bad else ''))
sys.exit(1 if bad else 0)
