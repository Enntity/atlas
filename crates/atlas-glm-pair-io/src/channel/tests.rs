// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn current() -> Credentials {
    unsafe {
        Credentials {
            pid: libc::getpid(),
            uid: libc::getuid(),
            gid: libc::getgid(),
        }
    }
}

#[test]
fn actual_packet_credentials_and_no_data_are_distinct() {
    let (left, right) = Channel::pair().unwrap();
    assert!(right.receive(current()).unwrap().is_none());
    assert!(left.send(b"actual authenticated child frame").unwrap());
    assert_eq!(
        right.receive(current()).unwrap().unwrap(),
        b"actual authenticated child frame"
    );
    assert!(right.receive(current()).unwrap().is_none());
}

#[test]
fn missing_inherited_descriptor_is_a_clean_error() {
    if std::env::var_os("ATLAS_PAIR_MISSING_FD_PROBE").is_some() {
        // Isolated CPU child owns this descriptor slot, with no Channel owner.
        unsafe {
            libc::close(3);
        }
        let result = unsafe { Channel::consume_inherited(3) };
        unsafe { libc::_exit(if result.is_err() { 0 } else { 91 }) }
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "channel::tests::missing_inherited_descriptor_is_a_clean_error",
            "--nocapture",
        ])
        .env("ATLAS_PAIR_MISSING_FD_PROBE", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn wrong_sender_and_eof_are_terminal() {
    let (left, right) = Channel::pair().unwrap();
    assert!(left.send(b"wrong sender").unwrap());
    let mut wrong = current();
    wrong.pid += 1;
    assert!(right.receive(wrong).is_err());
    drop(left);
    assert!(right.receive(current()).is_err());
}

fn send_rights(channel: &Channel, writer: RawFd, count: usize, payload: &[u8]) {
    // Aligned storage, including ample room for the large truncation probe.
    let mut control = [0usize; 64];
    let mut iov = libc::iovec {
        iov_base: payload.as_ptr().cast_mut().cast(),
        iov_len: payload.len(),
    };
    let mut message = unsafe { std::mem::zeroed::<libc::msghdr>() };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    message.msg_controllen =
        unsafe { libc::CMSG_SPACE((count * std::mem::size_of::<i32>()) as _) } as usize;
    assert!(message.msg_controllen <= std::mem::size_of_val(&control));
    unsafe {
        let header = libc::CMSG_FIRSTHDR(&message);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN((count * std::mem::size_of::<i32>()) as _) as usize;
        for index in 0..count {
            std::ptr::write_unaligned(libc::CMSG_DATA(header).cast::<i32>().add(index), writer);
        }
        assert_eq!(
            libc::sendmsg(channel.as_raw_fd(), &message, libc::MSG_NOSIGNAL),
            payload.len() as isize
        );
    }
}

#[test]
fn all_delivered_rights_are_closed_even_on_truncation_or_wrong_sender() {
    for (count, size, wrong_sender) in [
        (1, 1, false),
        (16, 1, false),
        (64, 1, false),
        (16, 4097, false),
        (16, 1, true),
        (16, 0, false),
    ] {
        let (left, right) = Channel::pair().unwrap();
        let mut fds = [-1; 2];
        assert_eq!(
            unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) },
            0
        );
        let reader = owned(fds[0]).unwrap();
        let writer = owned(fds[1]).unwrap();
        send_rights(&left, writer.as_raw_fd(), count, &vec![b'x'; size]);
        let mut expected = current();
        if wrong_sender {
            expected.pid += 1;
        }
        assert!(right.receive(expected).is_err());
        drop(writer);
        let mut byte = 0u8;
        // EOF proves that no received duplicate of the write end survived.
        assert_eq!(
            unsafe { libc::read(reader.as_raw_fd(), (&mut byte as *mut u8).cast(), 1) },
            0,
            "count={count}, size={size}, wrong={wrong_sender}"
        );
    }
}
