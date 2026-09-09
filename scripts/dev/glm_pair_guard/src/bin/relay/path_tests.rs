// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use std::os::unix::{fs::PermissionsExt, net::UnixListener};

#[test]
fn actual_private_directory_socket_and_symlink_checks() {
    let path = std::env::temp_dir().join(format!(
        "glm-relay-path-{}-{}",
        std::process::id(),
        Io::now().unwrap()
    ));
    std::fs::create_dir(&path).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    let name = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let uid = unsafe { libc::geteuid() };
    let gid = unsafe { libc::getegid() };
    let rank = directory(libc::AT_FDCWD, &name, uid, gid, true).unwrap();
    let socket_path = path.join("control.sock");
    let listener = UnixListener::bind(&socket_path).unwrap();
    std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let socket = connect_at(&rank, &crate::tests::config(), Io::now().unwrap(), uid, gid).unwrap();
    let (peer, _) = listener.accept().unwrap();
    drop((socket, peer));
    assert!(socket_stat(rank.as_raw_fd(), uid.wrapping_add(1), gid).is_err());
    std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o666)).unwrap();
    assert!(socket_stat(rank.as_raw_fd(), uid, gid).is_err());
    std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let real_socket = path.join("real.sock");
    std::fs::rename(&socket_path, &real_socket).unwrap();
    std::os::unix::fs::symlink("real.sock", &socket_path).unwrap();
    assert!(socket_stat(rank.as_raw_fd(), uid, gid).is_err());
    std::os::unix::fs::symlink(".", path.join("alias")).unwrap();
    assert!(directory(rank.as_raw_fd(), c"alias", uid, gid, true).is_err());
    std::fs::remove_file(path.join("alias")).unwrap();
    std::fs::remove_file(&socket_path).unwrap();
    std::fs::remove_file(&real_socket).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(directory(libc::AT_FDCWD, &name, uid, gid, true).is_err());
    drop((listener, rank));
    std::fs::remove_dir(&path).unwrap();
}
