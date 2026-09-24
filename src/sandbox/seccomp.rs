//! Syscall filter for agent processes. It denies what Docker's default profile denies to an
//! unprivileged container: creating namespaces, mounting, loading kernel code, and the
//! interfaces with the worst kernel exploit history. Everything else is allowed, so ordinary
//! programs behave as they do on a normal Linux host.

use anyhow::{Result, anyhow};
use seccompiler::{
    BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter, SeccompRule, TargetArch,
};
use std::collections::BTreeMap;

const NAMESPACE_FLAGS: [u64; 8] = [
    libc::CLONE_NEWUSER as u64,
    libc::CLONE_NEWNS as u64,
    libc::CLONE_NEWPID as u64,
    libc::CLONE_NEWNET as u64,
    libc::CLONE_NEWIPC as u64,
    libc::CLONE_NEWUTS as u64,
    libc::CLONE_NEWCGROUP as u64,
    0x80, // CLONE_NEWTIME
];

const DENIED: &[libc::c_long] = &[
    libc::SYS_unshare,
    libc::SYS_setns,
    libc::SYS_mount,
    libc::SYS_umount2,
    libc::SYS_pivot_root,
    libc::SYS_open_tree,
    libc::SYS_move_mount,
    libc::SYS_fsopen,
    libc::SYS_fsconfig,
    libc::SYS_fsmount,
    libc::SYS_fspick,
    libc::SYS_mount_setattr,
    libc::SYS_bpf,
    libc::SYS_perf_event_open,
    libc::SYS_io_uring_setup,
    libc::SYS_io_uring_enter,
    libc::SYS_io_uring_register,
    libc::SYS_keyctl,
    libc::SYS_add_key,
    libc::SYS_request_key,
    libc::SYS_userfaultfd,
    libc::SYS_kexec_load,
    libc::SYS_kexec_file_load,
    libc::SYS_init_module,
    libc::SYS_finit_module,
    libc::SYS_delete_module,
    libc::SYS_open_by_handle_at,
    libc::SYS_name_to_handle_at,
    libc::SYS_acct,
    libc::SYS_swapon,
    libc::SYS_swapoff,
    libc::SYS_reboot,
    libc::SYS_syslog,
    libc::SYS_quotactl,
    #[cfg(target_arch = "x86_64")]
    libc::SYS_iopl,
    #[cfg(target_arch = "x86_64")]
    libc::SYS_ioperm,
];

fn arch() -> Result<TargetArch> {
    std::env::consts::ARCH.try_into().map_err(|e| anyhow!("{e:?}"))
}

fn compile(rules: BTreeMap<i64, Vec<SeccompRule>>, errno: i32) -> Result<BpfProgram> {
    let filter = SeccompFilter::new(rules, SeccompAction::Allow, SeccompAction::Errno(errno as u32), arch()?)
        .map_err(|e| anyhow!("{e}"))?;
    filter.try_into().map_err(|e: seccompiler::BackendError| anyhow!("{e}"))
}

pub fn install() -> Result<()> {
    let mut denied: BTreeMap<i64, Vec<SeccompRule>> = DENIED.iter().map(|&n| (n, Vec::new())).collect();
    let namespace_clone = NAMESPACE_FLAGS
        .iter()
        .map(|&flag| {
            let condition = SeccompCondition::new(0, SeccompCmpArgLen::Qword, SeccompCmpOp::MaskedEq(flag), flag)
                .map_err(|e| anyhow!("{e}"))?;
            SeccompRule::new(vec![condition]).map_err(|e| anyhow!("{e}"))
        })
        .collect::<Result<Vec<_>>>()?;
    denied.insert(libc::SYS_clone, namespace_clone);
    // clone3 passes its flags through memory a filter cannot read. ENOSYS makes libc fall
    // back to clone, where the flags are visible.
    let clone3 = BTreeMap::from([(libc::SYS_clone3, Vec::new())]);
    for program in [compile(denied, libc::EPERM)?, compile(clone3, libc::ENOSYS)?] {
        seccompiler::apply_filter(&program).map_err(|e| anyhow!("{e}"))?;
    }
    Ok(())
}
