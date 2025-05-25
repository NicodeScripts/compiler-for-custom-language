import ply.lex as lex            #type:ignore
import ply.yacc as yacc          #type:ignore
from llvmlite import ir, binding #type:ignore
import platform, os, sys # If it doesn't work, maybe cuz it changed this, otherwise remove it

# ------------
#  Lexer code
# ------------
tokens = (
    'INT32', 'IDENTIFIER', 'NUMBER',
    'FUNC', 'RETURN', 'IF', 'SHOW', 'FOR', 'WHILE',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'COLON', 'SEMI', 'COMMA', 'ARROW',
    'PLUS', 'MINUS', 'MULT', 'DIV', 'MOD',
    'LT', 'GT', 'EQUAL','LOOPD'
)

# Token definitions
t_LPAREN   = r'\('
t_RPAREN   = r'\)'
t_LBRACE   = r'\{'
t_RBRACE   = r'\}'
t_COLON    = r':'
t_SEMI     = r';'
t_COMMA    = r','
t_ARROW    = r'>>'
t_PLUS     = r'\+'
t_MINUS    = r'-'
t_MULT     = r'\*'
t_DIV      = r'/'
t_MOD      = r'%'
t_LT       = r'<'
t_GT       = r'>'
t_EQUAL    = r'='

reserved = {
    'int32': 'INT32',
    'func': 'FUNC',
    'return': 'RETURN',
    'if': 'IF',
    'show': 'SHOW',
    'for': 'FOR',
    'while': 'WHILE',
    'loopd': 'LOOPD'
}

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

t_ignore = " \t\n"

def t_error(t):
    print(f"Illegal character {t.value[0]}")
    t.lexer.skip(1)

lexer = lex.lex()

# ----------------
#  Parser Ruleset
# ----------------

def p_program(p):
    '''program : declaration_list'''
    p[0] = p[1]

def p_declaration_list(p):
    '''declaration_list : declaration
                       | declaration declaration_list'''
    if len(p) == 2:
        p[0] = [p[1]] if p[1] is not None else []
    else:
        p[0] = [p[1]] + p[2] if p[1] is not None else p[2]

def p_declaration(p):
    '''declaration : var_declaration
                  | function_declaration'''
    p[0] = p[1]

def p_var_declaration(p):
    '''var_declaration : type IDENTIFIER COLON expression SEMI'''
    p[0] = ('var_decl', p[1], p[2], p[4])

def p_function_declaration(p):
    '''function_declaration : FUNC IDENTIFIER LPAREN func_args RPAREN ARROW type LBRACE statement_list RBRACE'''
    p[0] = ('function', p[2], p[4], p[7], p[9])

def p_func_args(p):
    '''func_args : arg_list
                | empty'''
    p[0] = p[1] if p[1] is not None else []

def p_arg_list(p):
    '''arg_list : type IDENTIFIER
                | type IDENTIFIER COMMA arg_list'''
    if len(p) == 3:
        p[0] = [(p[1], p[2])]
    else:
        p[0] = [(p[1], p[2])] + p[4]

def p_statement_list(p):
    '''statement_list : statement
                     | statement statement_list'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]

def p_statement(p):
    '''statement : var_declaration
                | return_statement
                | expression_statement
                | show_statement
                | loopd_statement'''
    p[0] = p[1]

def p_return_statement(p):
    '''return_statement : RETURN expression SEMI'''
    p[0] = ('return', p[2])

def p_expression_statement(p):
    '''expression_statement : expression SEMI'''
    p[0] = ('expr_stmt', p[1])

def p_show_statement(p):
    '''show_statement : SHOW LPAREN type RPAREN expression SEMI'''
    p[0] = ('show', p[3], p[5])

def p_expression(p):
    '''expression : term
                 | expression PLUS term
                 | expression MINUS term'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('binop', p[2], p[1], p[3])

def p_term(p):
    '''term : factor
            | term MULT factor
            | term DIV factor'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('binop', p[2], p[1], p[3])

def p_factor(p):
    '''factor : NUMBER
              | IDENTIFIER
              | LPAREN expression RPAREN
              | function_call'''
    if len(p) == 2:
        if isinstance(p[1], tuple):
            p[0] = p[1]
        elif isinstance(p[1], int):
            p[0] = ('number', p[1])
        else:
            p[0] = ('var', p[1])
    else:
        p[0] = p[2]

def p_function_call(p):
    '''function_call : IDENTIFIER LPAREN expression_list RPAREN'''
    p[0] = ('call', p[1], p[3])

def p_expression_list(p):
    '''expression_list : expression
                      | expression COMMA expression_list
                      | empty'''
    if len(p) == 2:
        if p[1] is None:
            p[0] = []
        else:
            p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]

def p_type(p):
    '''type : INT32'''
    p[0] = p[1]

def p_for_loop(p):
    '''statement : FOR LPAREN var_declaration expression SEMI expression RPAREN LBRACE statement_list RBRACE'''
    p[0] = ('for', p[3], p[4], p[6], p[9])

def p_while_loop(p):
    '''statement : WHILE LPAREN expression RPAREN LBRACE statement_list RBRACE'''
    p[0] = ('while', p[3], p[6])

def p_loopd_statement(p):
    '''loopd_statement : LOOPD LPAREN expression RPAREN LBRACE statement_list RBRACE'''
    p[0] = ('loopd', p[3], p[6])

def p_empty(p):
    '''empty :'''
    pass

def p_error(p):
    if p:
        print(f"Syntax error at token {p.type} with value '{p.value}' on line {p.lineno}")
    else:
        print("Syntax error at EOF")

parser = yacc.yacc()

# Get target triple based on current system cuz why not
def get_target_triple():
    system = platform.system()
    machine = platform.machine()
    
    # Map system to LLVM target
    if system == "Windows":
        os_name = "pc-windows-msvc"
    elif system == "Darwin":
        os_name = "apple-darwin"
    elif system == "Linux":
        os_name = "unknown-linux-gnu"
    else:
        os_name = "unknown-unknown"
    
    # Map architecture
    if machine in ["x86_64", "AMD64", "x64"]:
        arch = "x86_64"
    elif machine in ["i386", "i686", "x86"]:
        arch = "i386"
    elif machine in ["arm64", "aarch64"]:
        arch = "aarch64"
    elif machine.startswith("arm"):
        arch = "arm"
    else:
        arch = "unknown"
    
    return f"{arch}-{os_name}"

# LLVM IR Generation
def generate_llvm(ast, output_file):
    # Initialize LLVM
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()
    
    # Create module with the correct target triple
    target_triple = get_target_triple()
    module = ir.Module(name="IndigoModule")
    module.triple = target_triple
    
    # Add module-level variables and functions
    global_vars = {}
    functions = {}

    # Declare printf function (needed for the show statement)
    printf_type = ir.FunctionType(ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True)
    printf_func = ir.Function(module, printf_type, name="printf")
    functions["printf"] = printf_func
    
    # Helper function to evaluate expressions
    def eval_expr(expr, builder, local_vars):
        if expr[0] == 'number':
            return ir.Constant(ir.IntType(32), expr[1])
        elif expr[0] == 'var':
            var_name = expr[1]
            if var_name in local_vars:
                ptr = local_vars[var_name]
                if isinstance(ptr, ir.Instruction):
                    return builder.load(ptr)
                return ptr
            elif var_name in global_vars:
                return builder.load(global_vars[var_name])
            else:
                raise ValueError(f"Unknown variable: {var_name}")
        elif expr[0] == 'binop':
            op = expr[1]
            left = eval_expr(expr[2], builder, local_vars)
            right = eval_expr(expr[3], builder, local_vars)
            
            if op == '+':
                return builder.add(left, right)
            elif op == '-':
                return builder.sub(left, right)
            elif op == '*':
                return builder.mul(left, right)
            elif op == '/':
                return builder.sdiv(left, right)
            elif op == '%':
                return builder.srem(left, right)
        elif expr[0] == 'call':
            func_name = expr[1]
            args = expr[2]
            
            if func_name not in functions:
                raise ValueError(f"Unknown function: {func_name}")
            
            func = functions[func_name]
            evaluated_args = [eval_expr(arg, builder, local_vars) for arg in args]
            return builder.call(func, evaluated_args)
        
        raise ValueError(f"Unknown expression type: {expr[0]}")
    
    # First pass: Declare all functions and global variables
    for node in ast:
        if node[0] == 'var_decl':
            _, type_name, var_name, init_val = node
            if type_name == 'int32':
                gv = ir.GlobalVariable(module, ir.IntType(32), name=var_name)
                gv.linkage = 'internal'
                
                # For now, assume initializer is a simple number
                if init_val[0] == 'number':
                    gv.initializer = ir.Constant(ir.IntType(32), init_val[1])
                else:
                    # This will be handled in a second pass for more complex initializers
                    gv.initializer = ir.Constant(ir.IntType(32), 0)
                
                global_vars[var_name] = gv
                
        elif node[0] == 'function':
            name = node[1]
            args = node[2]
            return_type = node[3]
            
            # Create function type
            arg_types = [ir.IntType(32) for _ in args]
            func_type = ir.FunctionType(ir.IntType(32), arg_types)
            func = ir.Function(module, func_type, name=name)
            
            # Name the arguments
            for i, (_, arg_name) in enumerate(args):
                func.args[i].name = arg_name
                
            functions[name] = func
    
    # Second pass: Generate function bodies
    for node in ast:
        if node[0] == 'function':
            name = node[1]
            args = node[2]
            body = node[4]
            
            func = functions[name]
            block = func.append_basic_block(name="entry")
            builder = ir.IRBuilder(block)
            
            # Create a new scope for local variables
            local_vars = {}
            
            # Map function parameters to local variables
            for i, (_, arg_name) in enumerate(args):
                local_vars[arg_name] = func.args[i]
            
            # Process each statement in the function body
            for stmt in body:
                if stmt[0] == 'return':
                    expr = stmt[1]
                    ret_val = eval_expr(expr, builder, local_vars)
                    builder.ret(ret_val)
                elif stmt[0] == 'var_decl':
                    _, type_name, var_name, init_val = stmt
                    
                    if type_name == 'int32':
                        # Allocate space for the local variable
                        var_ptr = builder.alloca(ir.IntType(32), name=var_name)
                        
                        # Evaluate the initializer expression
                        init_value = eval_expr(init_val, builder, local_vars)
                        
                        # Store the initial value
                        builder.store(init_value, var_ptr)
                        
                        # Add the variable to the local scope
                        local_vars[var_name] = var_ptr
                elif stmt[0] == 'expr_stmt':
                    # Evaluate expression for side effects
                    eval_expr(stmt[1], builder, local_vars)
                elif stmt[0] == 'show':
                    # Handle show statement
                    _, type_name, expr = stmt
                    
                    # Evaluate the expression to be shown
                    value = eval_expr(expr, builder, local_vars)
                    
                    # Create appropriate format string based on type
                    if type_name == 'int32':
                        # Create the format string as a constant array directly in the function
                        fmt_str = ir.Constant(ir.ArrayType(ir.IntType(8), 4), bytearray("%d\n\0", 'utf8'))
                        
                        # Use alloca to create the format string on the stack
                        fmt_ptr_alloca = builder.alloca(fmt_str.type, name="fmt_str")
                        builder.store(fmt_str, fmt_ptr_alloca)
                        
                        # Bitcast to char* for printf
                        fmt_ptr = builder.bitcast(fmt_ptr_alloca, ir.PointerType(ir.IntType(8)))
                        
                        # Call printf with the formatted value
                        builder.call(printf_func, [fmt_ptr, value])  
                elif stmt[0] == 'for':
                        _, init, cond, update, body = stmt

                        # Create basic blocks for loop structure
                        loop_cond_block = builder.append_basic_block(name="loop_cond")
                        loop_body_block = builder.append_basic_block(name="loop_body")
                        loop_update_block = builder.append_basic_block(name="loop_update")
                        loop_exit_block = builder.append_basic_block(name="loop_exit")

                        # Process loop initialization
                        eval_expr(init, builder, local_vars)

                        # Jump to condition check
                        builder.branch(loop_cond_block)

                        # Condition check
                        builder.position_at_end(loop_cond_block)
                        cond_val = eval_expr(cond, builder, local_vars)
                        builder.cbranch(cond_val, loop_body_block, loop_exit_block)

                        # Loop body
                        builder.position_at_end(loop_body_block)
                        for stmt_body in body:
                            generate_statement(stmt_body, builder, local_vars) # FIX THIS
                        builder.branch(loop_update_block)
                        builder.position_at_end(loop_update_block)
                        eval_expr(update, builder, local_vars)
                        builder.branch(loop_cond_block)

                        # Loop exit
                        builder.position_at_end(loop_exit_block)
    # Add a terminator to functions that don't have one
    for func in module.functions:
        for block in func.blocks:
            if not block.is_terminated:
                ir.IRBuilder(block).ret(ir.Constant(ir.IntType(32), 0))

    # Verify LLVM module exists just in case
    try:
        binding.verify_module(module)
        print("✅ LLVM IR generation successful.")
    except Exception as e:
        print(f"❌ LLVM IR verification error: {e}, running anyway...")
    
    # Write LLVM IR to file
    with open(output_file, "w") as f:
        f.write(str(module))
    
    print(f"✅ LLVM IR written to {output_file}")

def parse(source_file, llvm_output):
    try:
        with open(source_file, "r") as f:
            code = f.read()

        print("🔍 Input Code:\n\n", code)

        ast = parser.parse(code)
        if ast is None:
            print("❌ Syntax error: Failed to parse code.")
            return False
        
        print("✅ AST Generated:", ast, "\n\nGenerating LLVM...")
        generate_llvm(ast, llvm_output)
        return True
    except Exception as e:
        print(f"❌ Error during compilation: {str(e)}")
        return False