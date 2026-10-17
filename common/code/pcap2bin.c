/* Convert a pcap on stdin to 12-byte records <uint32 ts_sec, uint32 src,
 * uint32 dst> (little-endian, address a.b.c.d as a<<24|b<<16|c<<8|d) for every
 * IPv4 packet. Handles Ethernet (with VLAN tags) and raw IP link types.
 *
 *   gcc -O2 -o pcap2bin pcap2bin.c
 *   gzip -dc trace.pcap.gz | ./pcap2bin > trace.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t sw32(uint32_t x) { return __builtin_bswap32(x); }

int main(void) {
    static unsigned char buf[262144];
    static uint32_t out[3 * 65536];
    size_t n_out = 0;
    uint32_t gh[6];
    if (fread(gh, 4, 6, stdin) != 6) { fprintf(stderr, "no pcap header\n"); return 1; }
    int swap;
    if (gh[0] == 0xa1b2c3d4 || gh[0] == 0xa1b23c4d) swap = 0;
    else if (gh[0] == 0xd4c3b2a1 || gh[0] == 0x4d3cb2a1) swap = 1;
    else { fprintf(stderr, "bad magic %08x\n", gh[0]); return 1; }
    uint32_t link = swap ? sw32(gh[5]) : gh[5];
    unsigned long long total = 0, ipv4 = 0, other = 0, short_pkt = 0;
    uint32_t first_ts = 0, last_ts = 0;
    for (;;) {
        uint32_t ph[4];
        if (fread(ph, 4, 4, stdin) != 4) break;
        uint32_t ts = swap ? sw32(ph[0]) : ph[0];
        uint32_t incl = swap ? sw32(ph[2]) : ph[2];
        if (incl > sizeof buf) { fprintf(stderr, "corrupt record (incl_len %u)\n", incl); return 1; }
        if (fread(buf, 1, incl, stdin) != incl) break;
        if (total == 0) first_ts = ts;
        last_ts = ts;
        total++;
        size_t off;
        if (link == 1) {                       /* Ethernet */
            if (incl < 14) { short_pkt++; continue; }
            uint16_t et = (uint16_t)(buf[12] << 8 | buf[13]);
            off = 14;
            while ((et == 0x8100 || et == 0x88a8 || et == 0x9100) && off + 4 <= incl) {
                et = (uint16_t)(buf[off + 2] << 8 | buf[off + 3]);
                off += 4;
            }
            if (et != 0x0800) { other++; continue; }
        } else if (link == 101 || link == 12 || link == 228) {
            off = 0;                           /* raw IP */
        } else { fprintf(stderr, "unsupported link type %u\n", link); return 1; }
        if (off + 20 > incl) { short_pkt++; continue; }
        if ((buf[off] >> 4) != 4) { other++; continue; }
        const unsigned char *s = buf + off + 12, *d = buf + off + 16;
        out[n_out++] = ts;
        out[n_out++] = (uint32_t)s[0] << 24 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 8 | s[3];
        out[n_out++] = (uint32_t)d[0] << 24 | (uint32_t)d[1] << 16 | (uint32_t)d[2] << 8 | d[3];
        ipv4++;
        if (n_out == sizeof out / sizeof out[0]) { fwrite(out, 4, n_out, stdout); n_out = 0; }
    }
    fwrite(out, 4, n_out, stdout);
    fprintf(stderr, "link %u: %llu packets, %llu IPv4, %llu other, %llu short; ts %u..%u\n",
            link, total, ipv4, other, short_pkt, first_ts, last_ts);
    return 0;
}
