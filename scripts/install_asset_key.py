"""Server-only: run the official licensed key installer, never expose its content."""
import os
from pathlib import Path
import socket


def main():
    if os.environ.get('ACCEPT_BEHAVIOR_DATASET_TERMS') != 'YES':
        raise SystemExit('Explicit asset license acceptance required')
    from cryptography.fernet import Fernet
    from omnigibson.utils import asset_utils
    path = Path(asset_utils.get_key_path())
    if path.exists():
        Fernet(path.read_bytes())
        path.chmod(0o600)
        print('ASSET_KEY_ALREADY_INSTALLED_AND_FORMAT_VALID', flush=True)
        return
    os.umask(0o077)
    socket.setdefaulttimeout(20)
    try:
        asset_utils.download_key()
        Fernet(path.read_bytes())
    except Exception as error:
        proxy = os.environ.get('ASSET_KEY_PROXY')
        if not proxy:
            raise
        print(f'Direct key download failed ({type(error).__name__}); retrying through temporary relay.', flush=True)
        # A failed urlretrieve may leave an empty file. Do not treat it as installed.
        if path.exists():
            try:
                Fernet(path.read_bytes())
            except (ValueError, TypeError):
                path.rename(path.with_suffix('.invalid-download'))
        import requests

        def retrieve(url, filename):
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(url, proxies={'http': proxy, 'https': proxy}, timeout=(20, 90))
                response.raise_for_status()
                content = response.content
                if len(content) > 1024:
                    raise ValueError('Unexpected key response size')
                Fernet(content)
                target = Path(filename)
                temporary = target.with_suffix('.download-part')
                temporary.write_bytes(content)
                temporary.chmod(0o600)
                temporary.replace(target)
                return str(target), response.headers

        asset_utils.urlretrieve = retrieve
        asset_utils.download_key()
        Fernet(path.read_bytes())
    path.chmod(0o600)
    print('ASSET_KEY_INSTALLED_AND_FORMAT_VALID', flush=True)


if __name__ == '__main__':
    main()
