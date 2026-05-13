import React, { useState } from 'react';
import { Paper, InputBase, IconButton, CircularProgress, Box } from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';

function CommandInput({ onExecuteCommand, isSubmitting }) {
    const [commandText, setCommandText] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!commandText.trim() || isSubmitting) return;

        const currentCommand = commandText;
        setCommandText('');
        await onExecuteCommand(currentCommand);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <Box sx={{ mb: 3 }}>
            <Paper
                component="form"
                onSubmit={handleSubmit}
                sx={{
                    p: '2px 4px',
                    display: 'flex',
                    alignItems: 'center',
                    width: '100%',
                    borderRadius: 2,
                    boxShadow: 3,
                    backgroundColor: 'background.paper',
                    border: '1px solid',
                    borderColor: 'divider'
                }}
            >
                <InputBase
                    sx={{ ml: 2, flex: 1, py: 1 }}
                    placeholder="Ask Catalyst Vector Alpha to do something..."
                    inputProps={{ 'aria-label': 'command input' }}
                    value={commandText}
                    onChange={(e) => setCommandText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={isSubmitting}
                    multiline
                    maxRows={4}
                />
                <IconButton
                    color="primary"
                    sx={{ p: '10px' }}
                    aria-label="submit command"
                    type="submit"
                    disabled={!commandText.trim() || isSubmitting}
                >
                    {isSubmitting ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
                </IconButton>
            </Paper>
        </Box>
    );
}

export default CommandInput;
