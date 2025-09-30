`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/30/2025 02:41:09 PM
// Design Name: blink_led_design
// Module Name: blink_led
// Project Name: blink_led
// Target Devices: Arty Z7
// Tool Versions: 
// Description: Test project
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


`default_nettype none
module blink_led #(
    parameter integer CNT_W = 26  // largura do contador; MSB define a taxa do blink
)(
    input  wire clk,    // ex.: FCLK_CLK0 do Zynq
    input  wire rst_n,  // reset ativo-baixo (ex.: FCLK_RESET0_N)
    output wire led
);
    // reset síncrono (sincroniza o rst_n para evitar metaestabilidade)
    reg [1:0] rst_sync;
    always @(posedge clk) begin
        rst_sync <= {rst_sync[0], rst_n};
    end
    wire rst = ~rst_sync[1]; // rst = 1 quando estiver em reset

    // contador livre: o bit mais significativo pisca sozinho
    reg [CNT_W-1:0] cnt_q;
    always @(posedge clk) begin
        if (rst) begin
            cnt_q <= {CNT_W{1'b0}};
        end else begin
            cnt_q <= cnt_q + {{(CNT_W-1){1'b0}}, 1'b1};
        end
    end

    assign led = cnt_q[CNT_W-1];  // usar o MSB como LED
endmodule
`default_nettype wire

