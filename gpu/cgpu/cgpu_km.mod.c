#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x4cf819e6, "module_layout" },
	{ 0x4a165127, "kobject_put" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0x696f0c3, "kmalloc_caches" },
	{ 0xd1769161, "kobject_get" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0xf09b5d9a, "get_zeroed_page" },
	{ 0x801e8735, "single_open" },
	{ 0x8e3e5524, "single_release" },
	{ 0x8a084f30, "no_llseek" },
	{ 0xa68165d, "seq_printf" },
	{ 0x837b7b09, "__dynamic_pr_debug" },
	{ 0x23295fe0, "__register_chrdev" },
	{ 0xb3f5e170, "filp_close" },
	{ 0x83599202, "seq_read" },
	{ 0x33a21a09, "pv_ops" },
	{ 0xd10a0a5f, "kthread_create_on_node" },
	{ 0xba223343, "nonseekable_open" },
	{ 0xefaeca87, "proc_remove" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0x456f0b3c, "PDE_DATA" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x119bae96, "kernel_read" },
	{ 0xddcba955, "follow_pfn" },
	{ 0xfb578fc5, "memset" },
	{ 0xf6897c7, "proc_mkdir" },
	{ 0xa22a96f7, "current_task" },
	{ 0xbcab6ee6, "sscanf" },
	{ 0x8a23e3a5, "kthread_stop" },
	{ 0xde80cd09, "ioremap" },
	{ 0x4c9d28b0, "phys_base" },
	{ 0x715a5ed0, "vprintk" },
	{ 0xfe487975, "init_wait_entry" },
	{ 0x800473f, "__cond_resched" },
	{ 0x7cd8d75e, "page_offset_base" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0x86b914bd, "module_put" },
	{ 0xd0da656b, "__stack_chk_fail" },
	{ 0x8ddd8aad, "schedule_timeout" },
	{ 0x92997ed8, "_printk" },
	{ 0x7f24de73, "jiffies_to_usecs" },
	{ 0x4a9a1936, "wake_up_process" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x4f00afd3, "kmem_cache_alloc_trace" },
	{ 0xba8fbd64, "_raw_spin_lock" },
	{ 0x4302d0eb, "free_pages" },
	{ 0xb3f7646e, "kthread_should_stop" },
	{ 0x8118637, "remove_proc_subtree" },
	{ 0x8c26d495, "prepare_to_wait_event" },
	{ 0xe240f6e9, "proc_create_data" },
	{ 0xa5457562, "seq_lseek" },
	{ 0x37a0cba, "kfree" },
	{ 0x43829e12, "unmap_mapping_range" },
	{ 0x69acdf38, "memcpy" },
	{ 0xedc03953, "iounmap" },
	{ 0x7d628444, "memcpy_fromio" },
	{ 0x556422b3, "ioremap_cache" },
	{ 0x92540fbf, "finish_wait" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0xb9314c99, "vmf_insert_pfn" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0x5c153fdc, "try_module_get" },
	{ 0xe914e41e, "strcpy" },
	{ 0xc4cc42af, "filp_open" },
	{ 0x81e6b37f, "dmi_get_system_info" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "BB5C1ECFDBE2550C4B2C779");
